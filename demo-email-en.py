import json
import logging
import os

import chromadb
import gradio as gr
import zhipuai
from operator import itemgetter
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

from llms.zhipuai_llm import ZhipuAILLM

_ = load_dotenv(find_dotenv())  # 读取并加载环境变量，来自 .env 文件

os.environ["http_proxy"] = os.environ["PROXY"]
os.environ["https_proxy"] = os.environ["PROXY"]
os.environ["no_proxy"] = os.environ["NO_PROXY"]
os.environ['CURL_CA_BUNDLE'] = ''

# 填写控制台中获取的 APIKey 信息
zhipuai.api_key = os.environ["ZHIPUAI_API_KEY"]

# Log setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "customer_enquiry_en"
PERSIST_DIRECTORY = "./database/hke/"
PATH_TO_SFT_JSON_FILES = './sft/'

CHROMA_CLIENT = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
CHROMA_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_RETRIEVER = None

# LLM Model
_LLM = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key=zhipuai.api_key, verbose=True)


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        # 加载文件夹中的所有txt类型的文件
        loader = DirectoryLoader('./data/en/', glob='**/*.txt')
        # 将数据转成 document 对象，每个文件会作为一个 document
        documents = loader.load()
        logger.info(f'documents:{len(documents)}')

        # 初始化加载器
        text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
        # 切割加载的 document
        split_docs = text_splitter.split_documents(documents)

        # RAG VectorSearch: 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        logger.info("building vector database index ...")
        embeddings = HuggingFaceEmbeddings(model_name=CHROMA_EMBEDDING_MODEL)
        if CHROMA_COLLECTION.count() > 0:
            vectorstore = Chroma(client=CHROMA_CLIENT,
                                 embedding_function=embeddings,
                                 collection_name=COLLECTION_NAME,
                                 persist_directory=PERSIST_DIRECTORY)
        else:
            vectorstore = Chroma.from_documents(split_docs,
                                                embedding=embeddings,
                                                collection_name=COLLECTION_NAME,
                                                persist_directory=PERSIST_DIRECTORY)
            vectorstore.persist()

        _RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 2})
    return _RETRIEVER


_TEMPLATE = """For the question provided below, check if any of the following criteria is matched. if any of the criteria matched, you should output yes. otherwise output no. Your output should use one word only.

Criteria:
1. Contain account information (e.g. account number)
2. Contain address or location information (e.g. relocate, move out)
3. Related to Account Operations (e.g. open account, setup account, close account, or terminate account)
4. Related to Bill (e.g. check bill, my billings, account balance, statement)
5. Related to Payment (e.g. fee, payment method, expenditure, price)
6. Related to Electricity Supply

Question: {question}

answer:"""

chain_precondition_check = LLMChain(
    llm=_LLM,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=_TEMPLATE
    ),
    output_key="output",
    verbose=True
)

_TEMPLATE = """For the input provided below, extract the address information if there is any, and then classified the address as Hong Kong Island, Kowloon, or New Territories. You should output in json format with keys in contain_address_info, address_classification, address_info.

Question: {question}

answer:"""

chain_location_check = LLMChain(
    llm=_LLM,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=_TEMPLATE
    ),
    output_key="output",
    verbose=True
)

_TEMPLATE = """

You are a customer service agent of HK electric, please help draft an email to  respond to the question at the end. 

You must follow the below rules in answering user question:

1. If the question is related to open or setup new account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US; 
2. If the question is related to close or terminate account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US; 
3. If the question is related to relocation or transfer account, first provide information to terminate account using use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US, and then provide information to setup new account using this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
4. If the question is related to deposit refund, it is preferred to use crossed cheque made payable; 
5. If the question is related to bill or statement, use this link: https://aol.hkelectric.com/AOL/aol#/login?lang=en-US


And additional information is provided in the below sections:
Customer Service Hotline or Phone Number:  (852) 2887 3411
Customer Service Email: cs@hkelectric.com
{context} 

Last, make your response precise and do not list all options, and it is always prefer to use electronic application form whenever applicable:

<Question> {question} </Question>

Helpful Answer:"""

chain_operation = (RunnableParallel({"documents": get_retriever(), "question": RunnablePassthrough()}) | {
    "documents": lambda x: [doc for doc in x["documents"]],
    "output": (
            {"context": lambda x: x["documents"], "question": itemgetter("question")}
            | PromptTemplate(input_variables=["context", "question"], template=_TEMPLATE)
            | _LLM
            | StrOutputParser()
    )
})

_TEMPLATE = """You are a customer service agent of HK electric, please help draft an email to response to greet the user politely, and inform the user that you are an email robot which fine tuned to customer service on account operation and billing enquiries and for Hong Kong Island region only. Because the enquiry is out of service scope, you have to redirect the customer to customer service by email to:	cs@hkelectric.com. Ask user do NOT reply this email since this is generated by AI system. Afterward, you should immediately end the conversation with user."""

chain_default = LLMChain(
    llm=_LLM,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=_TEMPLATE
    ),
    output_key="output",
    verbose=True
)


def route(info):
    logger.debug(info)
    is_valid = info["topic"]["output"]
    logger.info("chain_precondition_check: " + is_valid)
    if "yes" in is_valid.lower():
        chain = chain_location_check.invoke(info["topic"]["question"])
        out = json.loads(chain["output"].replace("\\n", "").replace("\\", "").replace("，", ","))
        logger.debug(out)
        logger.info("chain_address_check: " + str(out["contain_address_info"]))
        if out["contain_address_info"]:
            address_classification = out["address_classification"].lower()
            address_info = out["address_info"].lower()
            if "hong kong island" == address_classification:
                # serve Hong Kong Island
                chain = chain_operation.invoke(info["topic"]["question"])
                return chain
            else:
                # do not serve N.T and KLN
                chain = chain_default.invoke(info["topic"])
                return chain
        else:
            # do not contain address info
            chain = chain_operation.invoke(info["topic"]["question"])
        return chain
    else:
        chain = chain_default.invoke(info["topic"])
        return chain


full_chain = {"topic": chain_precondition_check} | RunnableLambda(
    route
)


def querying(query, history):
    result = full_chain.invoke(query)
    logger.info(result)
    return result["output"].strip().replace("\\n", "</br>").replace("\\", "")


def main():
    # Launch the interface
    # gr.ChatInterface(querying).launch(share=False)
    global _RETRIEVER
    _RETRIEVER = get_retriever()
    gr.ChatInterface(querying, title="This is an AI email assistant for customer service",
                     textbox=gr.Textbox(lines=10, scale=4)).launch(share=False, server_name="0.0.0.0", server_port=7863)



if __name__ == "__main__":
    main()

# INPUT = """
# Dear HK Electric,
#
# I'm planning to relocate to a new address: Room 908, 9/F WTT Tower, Causeway bay, Hong Kong next month.
# My existing account number is 0123456789  and I'm wondering what steps I need to take to ensure that I have
# electricity at my new home.
#
# Can you walk me through the process of applying for electricity supply, and let me know what information or
# documentation I'll need to provide?
#
# Thank you
#
# Regards,
# Mary
#
# """
