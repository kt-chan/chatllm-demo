import logging
import os
import json

import gradio as gr
import zhipuai
from dotenv import load_dotenv, find_dotenv
# Import langchain stuff
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from llms.zhipuai_llm import ZhipuAILLM
from langchain_core.runnables import RunnableLambda

_ = load_dotenv(find_dotenv())  # 读取并加载环境变量，来自 .env 文件

os.environ["http_proxy"] = os.environ["PROXY"]
os.environ["https_proxy"] = os.environ["PROXY"]
os.environ["no_proxy"] = os.environ["NO_PROXY"]
os.environ['CURL_CA_BUNDLE'] = ''

# 填写控制台中获取的 APIKey 信息
zhipuai.api_key = os.environ["ZHIPUAI_API_KEY"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_CUSTOMER_ENQUIRY = "customer_enquiry"
PERSIST_DIRECTORY = "./database/"
PATH_TO_SFT_JSON_FILES = './sft/'

# LLM Model
llm = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key=zhipuai.api_key)

TEMPLATE = """For the question provided below, check if any of the following criteria is matched. You should output 
yes with matched criteria or no only, and make your output in json format with key matched, criteria. 

Criteria:
1. Contain account information (e.g. account number)
2. Contain address or location information (e.g. relocate, move out)
3. Related to Account Operations (e.g. open account or close account)
4. Related to Billing (e.g. fees, usage, expenditure, price)
5. Related to electricity supply

Question: {question}

answer:"""

chain_precondition_check = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=TEMPLATE
    ),
    output_key="output",
    verbose=True
)

TEMPLATE = """For the input provided below, extract the address information if there is any, and then classified the address as Hong Kong Island, Kowloon, or New Territories.
You should output in json format with keys in contain_address_info, address_classification, address_info.

Question: {question}

answer:"""

chain_location_check = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=TEMPLATE
    ),
    output_key="output",
    verbose=True
)

TEMPLATE = """

You are a customer service agent of HK electric, please help draft an email to response to the 
question at the end. If you don't know the answer, just say "Hmm, I'm not sure.". Don't try to make up an answer.
You should follow the below rules in answering: 

1. Use electronic application whenever appropriate. 
2. For open account and transfer account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US; 
3. For close or terminate account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US; 
4. For  close or terminate account, you should also include deposit refund and prefer to use crossed cheque made payable; 
5. For relocation,  first provide information about account termination, and then provide information about account opening.

Question: {question}

Helpful Answer:"""

chain_operation = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=TEMPLATE
    ),
    output_key="output",
    verbose=True
)

TEMPLATE = """
Greet the user politely, and inform the user that you are an email robot which fine tuned to customer service on account 
operation and billing enquiries and for Hong Kong Island region only. And then redirect the customer to manual operation
by email to customer service email:	cs@hkelectric.com. Afterward, you should immediately end the conversation with user.
"""

chain_default = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=TEMPLATE
    ),
    output_key="output",
    verbose=True
)

INPUT = """
Dear HK Electric, 

I'm planning to relocate to a new address: Room 908, 9/F WTT Tower, Yuen Long, Hong Kong next month. 
My existing account number is 0123456789  and I'm wondering what steps I need to take to ensure that I have 
electricity at my new home.

Can you walk me through the process of applying for electricity supply, and let me know what information or 
documentation I'll need to provide? 

Thank you

Regards,
Mary

"""


def route(info):
    logging.debug(info)
    out = json.loads(info["topic"]["output"].replace("\\n", "").replace("\\", ""))
    logging.info("chain_precondition_check: " + out["matched"])
    if "yes" in out["matched"].lower():
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
            logging.debug(chain)
        return chain
    else:
        chain = chain_default
        logging.debug(chain)
        return chain


full_chain = {"topic": chain_precondition_check, "question": lambda x: x["question"]} | RunnableLambda(
    route
)

output = full_chain.invoke({"question": INPUT})
logging.info(output)

# # print(overall_chain({"input":input}))
# class QAPair:
#     def __init__(self, question, answer):
#         self.question = question
#         self.answer = answer
#
#
# # 加载文件夹中的所有txt类型的文件
# loader = DirectoryLoader('./data/', glob='**/*.txt')
# # 将数据转成 document 对象，每个文件会作为一个 document
# documents = loader.load()
# logger.info(f'documents:{len(documents)}')
#
# # 初始化加载器
# text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
# # 切割加载的 document
# split_docs = text_splitter.split_documents(documents)
#
# # RAG VectorSearch: 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
# vector_search = Chroma.from_documents(split_docs,
#                                       embedding=HuggingFaceEmbeddings(),
#                                       collection_name=COLLECTION_CUSTOMER_ENQUIRY,
#                                       persist_directory=PERSIST_DIRECTORY
#                                       )
#
# custom_question_prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=TEMPLATE)
#
# # 定义内存记忆
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# def querying(query, history):
#     overall_chain = SequentialChain(
#         chains=[chain1, chain2, chain3],
#         input_variables=["input"],
#         output_variables=["answer"],
#         #memory=["chat_history"],
#         verbose=True
#     )
#     result = overall_chain({"input": query, "chat_history": history})
#     logger.info("memory:")
#     logger.info(memory.chat_memory.messages)
#     logger.info("question: " + query)
#     logger.info(result)
#     return result["answer"].strip().replace("\\n", "</br>")
#
#
# # Launch the interface
# # gr.ChatInterface(querying).launch(share=False)
# gr.ChatInterface(querying, title="This is an AI email assistant for customer service",
#                  textbox=gr.Textbox(lines=10, scale=4)).launch(share=False)
