import logging
import os

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
from langchain_core.prompts import PromptTemplate

from llms.zhipuai_llm import ZhipuAILLM

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

# INSTRUCTION = """You are a customer service agent of HK electric, please respond to the question at the end. If the question is not related account operation or billing enquiries, you have to decline answering and politely inform the user that you are only tuned to customer service on account operation and billing enquiries.
#
# You must follow the below rules in answering user question:
#
# 1. If the question is related to open or setup new account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US;
# 2. If the question is related to close or terminate account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
# 3. If the question is related to relocation or transfer account, first provide information to terminate account using use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US, and then provide information to setup new account using this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
# 4. If the question is related to deposit refund, it is preferred to use crossed cheque made payable;
# 5. If the question is related to bill or statement, use this link: https://aol.hkelectric.com/AOL/aol#/login?lang=en-US
#
#
# Last, make your response precise and do not list all options, and it is always prefer to use electronic application form whenever applicable:
#
#
# Question: """

RAG_TEMPLATE = """You are a customer service agent of HK electric, please respond to the question at the end. If the question is not related account operation or billing enquiries, you have to decline answering and politely inform the user that you are only tuned to customer service on account operation and billing enquiries.
You must follow the below rules in answering user question:
1. If the question is related to open or setup new account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US; 
2. If the question is related to close or terminate account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US; 
3. If the question is related to relocation or transfer account, first provide information to terminate account using use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US, and then provide information to setup new account using this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
4. If the question is related to deposit refund, it is preferred to use crossed cheque made payable; 
5. If the question is related to bill or statement, use this link: https://aol.hkelectric.com/AOL/aol#/login?lang=en-US

And additional information is provided in the below sections:
{context}

Chat history:
{chat_history}

Question: {question}

Helpful Answer:"""


class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('./data/', glob='**/*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()
logger.info(f'documents:{len(documents)}')

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# LLM Model
llm = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key=zhipuai.api_key)

# RAG VectorSearch: 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
vector_search = Chroma.from_documents(split_docs,
                                      embedding=HuggingFaceEmbeddings(),
                                      collection_name=COLLECTION_CUSTOMER_ENQUIRY,
                                      persist_directory=PERSIST_DIRECTORY
                                      )

custom_question_prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=RAG_TEMPLATE)


def querying(query, history):
    # 定义内存记忆
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if history:
        logger.debug("chat history:")
        logger.debug(history)
        for itemset in history:
            logger.debug("input:" + itemset[0] + "; output: " + itemset[1])
            msg_human = itemset[0]
            msg_bot = itemset[1]
            memory.save_context({"input": msg_human}, {"output": msg_bot})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_search.as_retriever(search_kwargs={"k": 1}),
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": custom_question_prompt}
    )
    logger.info("memory:")
    logger.debug(memory.chat_memory.messages)
    logger.debug("question: " + query)

    result = qa_chain({"question": query})
    logger.debug("answer: " + result["answer"].strip())
    return result["answer"].strip().replace("\\n", "</br>")


# Launch the interface
# gr.ChatInterface(querying).launch(share=False)
gr.ChatInterface(querying, title="This is an AI chatbot for customer service").launch(share=False,
                                                                                      server_name="0.0.0.0",
                                                                                      server_port=7862)
