import logging
import os
import pprint
import uuid
from typing import List

import chromadb
import gradio as gr
import requests
import zhipuai
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
# Import langchain stuff
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llms.zhipuai_llm import ZhipuAILLM
from langchain.chains import create_extraction_chain

_ = load_dotenv(find_dotenv())  # 读取并加载环境变量，来自 .env 文件

os.environ["http_proxy"] = os.environ["PROXY"]
os.environ["https_proxy"] = os.environ["PROXY"]
os.environ["no_proxy"] = os.environ["NO_PROXY"]
os.environ['CURL_CA_BUNDLE'] = ''

# 填写控制台中获取的 APIKey 信息
zhipuai.api_key = os.environ["ZHIPUAI_API_KEY"]
# # LLM Model
llm = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key=zhipuai.api_key)

# Log setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "webfaq_en"
PERSIST_DIRECTORY = "./database/cncbi/en/"
PATH_TO_SFT_JSON_FILES = './sft/'
REF_WEBSITE_LINK = ["https://www.cncbinternational.com/personal/e-banking/inmotion/en/support/index.html"]

CHROMA_CLIENT = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
CHROMA_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # or use multilingual sentence-transformers/LaBSE

RAG_TEMPLATE = """You are a customer service agent of China CITIC Bank International, and please respond to the question at the end. If the question is not related to the bank's customer service, you have to decline answering and politely inform the user that you are only tuned to bank customer service. Do not make up the answer from your general knowledge, and if you cannot find reference information from the below Frequently Asked Questions and Answers, just refer the customer to the customer hotline at 22876767.

Frequently Asked Questions and Answers:
{context}

Chat history:
{chat_history}

Question: {question}

Helpful Answer:"""


class QAPair:
    def __init__(self, question, answers):
        self.question = question
        self.answers = answers

    def __str__(self):
        return f'question: {self.question} , answers: {"; ".join(self.answers)}'


def scrape_webpages(urls):
    faq_listings = {}
    for url in urls:
        logger.info("fetching page " + url)
        loader = requests.get(url)
        soup = BeautifulSoup(loader.content, 'html.parser')

        q_listings = {}
        a_listings = {}
        qa_listings = {}

        faq_content = soup.find('div', class_='faq-contain')
        logger.debug("faq_content")
        logger.debug(faq_content)
        q_items = faq_content.find_all(class_='faq-question-wrapper')
        a_items = faq_content.find_all(class_='faq-answer-wrapper')

        k = 0
        for q_item in q_items:
            logger.debug("q_item on key = " + str(k))
            logger.debug(q_item)
            questions = q_item.find_all('p')
            for question in questions:
                if len(question.text.strip()) > 0:
                    q_listings.setdefault(k, []).append(question.text.strip())
            k = k + 1

        k = 0
        for a_item in a_items:
            logger.debug("a_item on key = " + str(k))
            logger.debug(a_item)
            answers = a_item.find_all(['p', 'li'])
            for answer in answers:
                if len(answer.text.strip()) > 0:
                    a_listings.setdefault(k, []).append(answer.text.strip())
            k = k + 1

        for q in q_listings:
            qa_listings[q] = {(tuple(q_listings[q]), tuple(a_listings[q]))}

        logger.debug(qa_listings)
        faq_listings.setdefault(url, []).append(qa_listings)
    return faq_listings


# extracted_content = scrape_with_playwright(REF_WEBSITE_LINK)
# logger.info(extracted_content)

def extract_docs(urls):
    my_docs: List[Document] = list()
    for k, v in scrape_webpages(urls).items():
        logger.info("parsing page " + k)
        for doc in v:
            for pair in doc:
                questions = list(doc[pair])[0][0][0]
                answers = list(doc[pair])[0][1]
                qa_pair = QAPair(questions.strip(), answers)
                my_docs.append(Document(page_content=str(qa_pair), metadata={"source": k}))
    return my_docs


# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(extract_docs(REF_WEBSITE_LINK))

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
                                        persist_directory=PERSIST_DIRECTORY
                                        )
    vectorstore.persist()

chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
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
        retriever=chroma_retriever,
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
                                                                                      server_port=7864)
