import logging
import os

import chromadb
import gradio as gr
import zhipuai
from dotenv import load_dotenv, find_dotenv
# Import langchain stuff
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
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

COLLECTION_NAME = "customer_enquiry_en"
PERSIST_DIRECTORY = "./database/hke/"
PATH_TO_SFT_JSON_FILES = './sft/'

CHROMA_CLIENT = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
CHROMA_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # this is for english embeddings or use multilingual sentence-transformers/LaBSE
# CHROMA_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese-paraphrase" # this is chinese embeddings.

RAG_TEMPLATE = """You are a customer service agent of HK electric, please respond to the question at the end precisely with less than 5 lines of text. 
You must follow the below rules in answering user question:
1. If the question is related to open or setup new account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US; 
2. If the question is related to close or terminate account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US; 
3. If the question is related to relocation or transfer account, first provide information to terminate account using use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US, and then provide information to setup new account using this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
4. If the question is related to deposit refund, it is preferred to use crossed cheque made payable; 
5. If the question is related to bill or statement, use this link: https://aol.hkelectric.com/AOL/aol#/login?lang=en-US

Reference Information:
{context}

Chat history:
{chat_history}

Question: {question}

Helpful Answer:"""

VALIDATION_TEMPLATE = """For the question provided below, check if any of the following criteria is matched. if any of the criteria matched, you should output yes. otherwise output no. Your output should use one word only.

Criteria:
1. Contain account information (e.g. account number)
2. Contain address or location information (e.g. relocate, move out)
3. Related to Account Operations (e.g. open account, setup account, close account, or terminate account)
4. Related to Bill (e.g. check bill, my billings, account balance, statement)
5. Related to Payment (e.g. fee, payment method, expenditure, price)
6. Related to Electricity Supply

Question: {question}

answer:"""
class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('./data/en/', glob='**/*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()
logger.info(f'documents:{len(documents)}')

# 初始化加载器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
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

#chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
chroma_retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1, "k": 1})
custom_question_prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=RAG_TEMPLATE)

chain_precondition_check = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=VALIDATION_TEMPLATE
    ),
    output_key="output",
    verbose=True
)

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


    valid_output = chain_precondition_check.invoke(query)
    logger.info("chain_precondition_check: ")
    is_valid = valid_output['output']
    logger.info(is_valid.lower())

    output = "This is AI chatbot for customer service on account operation and bill enquiry, for other question please call our customer service hotline at +852 2887 3466."
    if "yes" in is_valid.lower():
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
        output = result["answer"].strip().replace("\\n", "</br>")

    return output


# Launch the interface
# gr.ChatInterface(querying).launch(share=False)
gr.ChatInterface(querying, title="This is an AI chatbot for customer service").launch(share=False,
                                                                                      server_name="0.0.0.0",
                                                                                      server_port=7861)
