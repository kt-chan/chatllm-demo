import json
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
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

from langchain.chains import LLMChain
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

INSTRUCTION = """You are a customer service agent of HK electric, please help draft an email to response to the 
question at the end. If you don't know the answer, just say "Hmm, I'm not sure.". Don't try to make up an answer. If 
the question is not related account operation or billing enquiries, you have to decline answering and politely inform 
the user that you are only tuned to customer service on account operation and billing enquiries. You should follow 
the below rules in answering: 
1. Use electronic application whenever appropriate. 
2. For open account and transfer account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US; 
3. For close or terminate account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US; 
4. For  close or terminate account, you should also include deposit refund and prefer to use crossed cheque made payable; 
5. For relocation,  first provide information about account termination, and then provide information about account openning.

Question: """

PREFIX_TEMPLATE = """You are a customer service agent of HK electric who answer questions from customer enquiry. Use the 
following pieces of context to answer the question at the end. If you don't know the answer, just say "Hmm, 
I'm not sure.". Don't try to make up an answer. If the question is not related account operation or billing 
enquiries, you have to decline answering and politely inform the user that you are only tuned to customer service 
on account operation and billing enquiries. The following examples are provided for your reference:"""

SUFFIX_TEMPLATE = """
Question: {question}
"""


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

# 定义内存记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# 将数据转成 document 对象，每个文件会作为一个 document
def getExampleJson():
    json_file_names = [filename for filename in os.listdir(PATH_TO_SFT_JSON_FILES) if filename.endswith('.json')]
    for json_file_name in json_file_names:
        with open(os.path.join(PATH_TO_SFT_JSON_FILES, json_file_name), encoding="utf8") as json_file:
            return json.load(json_file)
            # for json_obj in json_objs:
            #     memory.chat_memory.add_user_message(re.sub(' +', ' ', json_obj['question']))
            #     memory.chat_memory.add_ai_message(re.sub(' +', ' ',  json_obj['answer']))


# Prompt-Template

# custom_question_prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=RAG_TEMPLATE)
example_template = """Question: {question}\n{answer}"""

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_template
)

custom_question_prompt = FewShotPromptTemplate(
    examples=getExampleJson(),
    example_prompt=example_prompt,
    prefix=PREFIX_TEMPLATE,
    suffix="Question: {question}",
    input_variables=["question"]
)


def querying(query, history):
    chain = LLMChain(llm=llm, prompt=custom_question_prompt)

    logger.info("memory:")
    logger.info(memory.chat_memory.messages)
    logger.info("question: " + INSTRUCTION + query)

    result = chain.invoke({"question": INSTRUCTION + query})
    logger.info("answer: " + result["text"].strip())
    return result["text"].strip().replace("\\n", "</br>")


# Launch the interface
# gr.ChatInterface(querying).launch(share=False)
gr.ChatInterface(querying, title="This is an AI email assistant for customer service",
                 textbox=gr.Textbox(lines=10, scale=4)).launch(share=False)
