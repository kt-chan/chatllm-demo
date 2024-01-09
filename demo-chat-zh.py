import logging
import os

import chromadb
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

# Log setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "customer_enquiry_zh"
PERSIST_DIRECTORY = "./database/hke/"
PATH_TO_SFT_JSON_FILES = './sft/'

CHROMA_CLIENT = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
# CHROMA_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # this is for english embeddings or use multilingual sentence-transformers/LaBSE
CHROMA_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese-paraphrase" # this is chinese embeddings.

RAG_TEMPLATE = """你是HK Electric的客服代表，請在回答本信息最後部分的問題。如果問題與賬戶操作或賬單查詢無關，您必須拒絕回答，幷禮貌地告知用戶您只能處理客服處理賬戶操作和賬單查詢。
回答用戶問題時，必須遵循以下原則:
1. 如果問題與開設或設置新帳戶有關，請使用此鏈接:https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US;
2. 如果問題與關閉或終止帳戶有關，請使用此鏈接:https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
3. 如果問題涉及搬遷或轉讓賬戶，請先提供信息使用此鏈接:https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US終止賬戶，然後提供信息使用此鏈接:https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US建立新賬戶;
4. 如果問題是有關退還押金，最好使用劃綫支票付款;
5. 如果問題與賬單或報表有關，請使用此鏈接:https://aol.hkelectric.com/AOL/aol#/login?lang=en-US

以下部分提供了更多信息供你參考:
{context}

聊天记录:
{chat_history}

问题: {question}

最終回答:"""


class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('./data/zh/', glob='**/*.txt')
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

chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
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
gr.ChatInterface(querying, title="這是一個客戶服務的人工智能聊天機器人").launch(share=False,
                                                                                      server_name="0.0.0.0",
                                                                                      server_port=7862)
