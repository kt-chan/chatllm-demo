import json
import logging
import os

import gradio as gr
import zhipuai
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnableLambda

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
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

_COLLECTION_CUSTOMER_ENQUIRY = "customer_enquiry"
_PERSIST_DIRECTORY = "./database/"
_PATH_TO_SFT_JSON_FILES = './sft/'
_RETRIEVER = None
_MEMORY = None

# LLM Model
_LLM = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key=zhipuai.api_key)

_TEMPLATE = """For the question provided below, check if any of the following criteria is matched. if any of the 
criteria matched, you should output yes. otherwise output no. Your output should use one word only.

Criteria:
1. Contain account information (e.g. account number)
2. Contain address or location information (e.g. relocate, move out)
3. Related to Account Operations (e.g. open account, setup account, close account, or terminate account)
4. Related to Billing (e.g. fees, usage, expenditure, price)
5. Related to electricity supply

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

_TEMPLATE = """For the input provided below, extract the address information if there is any, and then classified the address as Hong Kong Island, Kowloon, or New Territories.
You should output in json format with keys in contain_address_info, address_classification, address_info.

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

You are a customer service agent of HK electric, please help draft an email to response to the 
question at the end. If you don't know the answer, just say "Hmm, I'm not sure.". Don't try to make up an answer.
You should follow the below rules in answering: 

1. Use electronic application whenever appropriate. 
2. For open account and transfer account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US; 
3. For close or terminate account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US; 
4. For  close or terminate account, you should also include deposit refund and prefer to use crossed cheque made payable; 
5. For relocation,  first provide information about account termination, and then provide information about account opening.

You should take below context information for reference.
Context: 
{context}

You should also check the following chat history for reference.
Chat history:
{chat_history}


Question: {question}

Helpful Answer:"""


def getRetriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        # 加载文件夹中的所有txt类型的文件
        loader = DirectoryLoader('./data/', glob='**/*.txt')
        # 将数据转成 document 对象，每个文件会作为一个 document
        documents = loader.load()
        logger.info(f'documents:{len(documents)}')

        # 初始化加载器
        text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
        # 切割加载的 document
        split_docs = text_splitter.split_documents(documents)

        # RAG VectorSearch: 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        vector_store = Chroma.from_documents(split_docs,
                                             embedding=HuggingFaceEmbeddings(),
                                             collection_name=_COLLECTION_CUSTOMER_ENQUIRY,
                                             persist_directory=_PERSIST_DIRECTORY
                                             )
        _RETRIEVER = vector_store.as_retriever(search_kwargs=dict(k=2))
    return _RETRIEVER


def getMemory():
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return _MEMORY


chain_operation = ConversationalRetrievalChain.from_llm(
    llm=_LLM,
    retriever=getRetriever(),
    memory=getMemory(),
    combine_docs_chain_kwargs={"prompt": PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=_TEMPLATE
    )},
    output_key="output",
    verbose=True
)

_TEMPLATE = """You are a customer service agent of HK electric, please help draft an email to response to greet the 
user politely, and inform the user that you are an email robot which fine tuned to customer service on account 
operation and billing enquiries and for Hong Kong Island region only. Because the enquiry is out of service scope, 
you have to redirect the customer to customer service by email to:	cs@hkelectric.com. Afterward, you should 
immediately end the conversation with user."""

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
    logging.debug(info)
    isValid = info["topic"]["output"]
    logging.info("chain_precondition_check: " + isValid)
    if "yes" in isValid.lower():
        chain = chain_location_check({"question": info["question"]})
        out = json.loads(chain["output"].replace("\\n", "").replace("\\", ""))
        logging.debug(out)
        logging.info("chain_address_check: " + str(out["contain_address_info"]))
        if out["contain_address_info"]:
            address_classification = out["address_classification"].lower()
            address_info = out["address_info"].lower()
            if "hong kong island" == address_classification:
                # serve Hong Kong Island
                chain = chain_operation({"question": info["question"]})
                logging.debug(getMemory().chat_memory.messages)
                logging.debug(chain)
                return chain
            else:
                # do not serve N.T and KLN
                chain = chain_default
                logging.debug(chain)
                return chain
        else:
            # do not contain address info
            chain = chain_operation({"question": info["question"]})
            logging.debug(getMemory().chat_memory.messages)
            logging.debug(chain)
        return chain
    else:
        chain = chain_default
        logging.debug(chain)
        return chain


full_chain = {"topic": chain_precondition_check, "question": lambda x: x["question"]} | RunnableLambda(
    route
)


def querying(query, history):
    result = full_chain.invoke({"question": query})
    logging.info(result)
    return result["output"].strip().replace("\\n", "</br>").replace("\\", "")


def main():
    # Launch the interface
    # gr.ChatInterface(querying).launch(share=False)
    global _RETRIEVER, _MEMORY
    _RETRIEVER = getRetriever()
    _MEMORY = getMemory()
    gr.ChatInterface(querying, title="This is an AI email assistant for customer service",
                     textbox=gr.Textbox(lines=10, scale=4)).launch(share=False)


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
