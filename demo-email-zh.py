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

COLLECTION_NAME = "customer_enquiry_zh"
PERSIST_DIRECTORY = "./database/hke/"
PATH_TO_SFT_JSON_FILES = './sft/'

CHROMA_CLIENT = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
CHROMA_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese-paraphrase" # this is chinese embeddings.
_RETRIEVER = None

# LLM Model
_LLM = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key=zhipuai.api_key, verbose=True)


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        # 加载文件夹中的所有txt类型的文件
        loader = DirectoryLoader('./data/zh/', glob='**/*.txt')
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


_TEMPLATE = """對于下面提供的問題，檢查是否符合以下任何條件。如果匹配任何條件，您應該輸出yes。否則輸出no。您的輸出應該只使用一個單詞。
條件:
1. 包含賬戶信息(如賬號)
2. 包含地址或位置信息(例如搬遷、搬遷)
3. 與客戶操作相關的工作(如開立賬戶、設立賬戶、關閉賬戶或終止賬戶)
4. 與賬單相關(如支票賬單、我的賬單、賬戶餘額、對賬單)
5. 有關付款(如費用、付款方式、自動轉賬、電費、價格)
6. 有關電力供應


問題: {question}

回答:"""

chain_precondition_check = LLMChain(
    llm=_LLM,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=_TEMPLATE
    ),
    output_key="output",
    verbose=True
)

_TEMPLATE = """對于下面提供的輸入，提取地址信息(如果有)，然後將地址分類爲香港島、九龍或新界。您應該以json格式輸出，其中json的key爲contain_address_info, address_classification, address_info.

問題: {question}

回答:"""

chain_location_check = LLMChain(
    llm=_LLM,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=_TEMPLATE
    ),
    output_key="output",
    verbose=True
)

_TEMPLATE = """你是HK Electric的客服代理，請幫忙起草一封郵件回復本信息最後部份的問題。
回答用戶問題時，必須遵循以下原則:
1. 如果問題與開設或設置新帳戶有關，請使用此鏈接:https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=zh;
2. 如果問題與關閉或終止帳戶有關，請使用此鏈接:https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=zh;
3. 如果問題涉及搬遷或轉讓賬戶，請先提供信息使用此鏈接終止賬戶:https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=zh，然後提供信息使用此鏈接建立新賬戶:https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=zh;
4. 如果問題是有關退還押金，最好使用劃綫支票付款;
5. 如果問題與賬單或報表有關，請使用此鏈接:https://aol.hkelectric.com/AOL/aol#/login?lang=zh

以下部分提供了更多信息供你參考:
{context} 

最後，你的回答要準確，不要列出所有的選項，幷且在適用的情况下，總是傾向于使用電子申請表格.

問題: {question} 

最終回答:"""

chain_operation = (RunnableParallel({"documents": get_retriever(), "question": RunnablePassthrough()}) | {
    "documents": lambda x: [doc for doc in x["documents"]],
    "output": (
            {"context": lambda x: x["documents"], "question": itemgetter("question")}
            | PromptTemplate(input_variables=["context", "question"], template=_TEMPLATE)
            | _LLM
            | StrOutputParser()
    )
})

_TEMPLATE = """你是香港電燈的客戶服務代表，請幫忙起草一封回復郵件，禮貌地問候用戶，幷通知用戶你是一個電子郵件機器人，只針對港島地區的賬戶操作和賬單查詢提供客戶服務。由于查詢超出了服務範圍，您必須通過郵件將客戶重定向到客服:cs@hkelectric.com。請用戶不要回復此郵件，因爲這是由人工智能系統生成的。之後，您應該立即結束與用戶的對話。"""

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
            if "hong kong island" == address_classification or "香港島" == address_classification:
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
    gr.ChatInterface(querying, title="這是一個客戶服務的人工智能電子郵件助手",
                     textbox=gr.Textbox(lines=10, scale=4)).launch(share=False, server_name="0.0.0.0", server_port=7864)



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
