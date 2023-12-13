import os, logging
import zhipuai
import gradio as gr
import re
# Import langchain stuff
from langchain import hub
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from llms.zhipuai_llm import ZhipuAILLM
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # 读取并加载环境变量，来自 .env 文件

os.environ["http_proxy"] = os.environ["PROXY"]
os.environ["https_proxy"] = os.environ["PROXY"]
os.environ["no_proxy"] = os.environ["NO_PROXY"]

# 填写控制台中获取的 APIKey 信息
zhipuai.api_key = os.environ["ZHIPUAI_API_KEY"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INSTRUCTION = """You are a customer service agent of HK electric, please help draft an email to response to the question at the end.
You should follow the below rules in answering:
1. Use electronic application whenever appropriate.
2. For open account and transfer account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US;
3. For close account, use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
4. For account termination or account close, you have to provide information about deposit refund and it is preferred to use crossed cheque made payable;
5. For relocation,  first provide information about account termination, and then provide information about account openning.

Question: """


RAG_TEMPLATE = """You are a customer service agent of HK electric who answer questions from customer enquiry. 
Use the following pieces of context to answer the question at the end.  
If you don't know the answer, just say "Hmm, I'm not sure.". Don't try to make up an answer. 
If the question is not about customer service scope, politely inform them that you are only tuned to customer service.

{context}

Chat history:
{chat_history}

Question: {question}

Helpful Answer:"""


class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


samples = [
    QAPair("""
        Dear HK Electric, 

        I'm planning to relocate to a new address: Room 1001, 10/F ABC Building, 100 Oil Street, North Point, HK next month.  My existing account number is 0123456789  and I'm wondering what steps I need to take to ensure that I have electricity at my new home. Can you walk me through the process of applying for electricity supply, and let me know what information or documentation I'll need to provide? Thank you

        Regards,
        Mary
        """,
     """
    Dear Mary, 

    Service address: Room 1001, 10/F ABC Building, 100 Oil Street, North Point, HK

    Thank you for your email.

    For moving house, the registered customer has to arrange termination of the current electricity account and set up a new one for the new address. Would you please advise the registered customer to contact our Customer Services Executives at 2887 3411 during office hours or complete the electronic forms at the path below to process the requests: 

    Account Termination (with at least 2 working days’ advance notice and should not fall on Saturday / Sunday / Public Holiday)
    https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US

    For refund of the deposit balance (after deducting the final outstanding as at the termination date from the deposit), we could send a crossed cheque in Hong Kong Dollars payable to the registered customer name to the new correspondence address in around 5 working days upon closure of account. 

    Account Registration (with at least 1 working day advance notice and should not fall on Saturday / Sunday / Public Holiday)
    https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US

    Upon successful registration, a deposit is required as security for future use of electricity and a new electricity account number will be assigned. The electricity supply will be connected during office hours on the account effective date. 

    If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours. 
    Yours sincerely,
    YM Lai
    Senior Manager (Customer Services)
    HK Electric
    """),
    QAPair("""
        Hi CS team, 

        My account is 0123456789.  
        I will move to a new flat at the end of this month. 
        Should I transfer my account to the new apartment or should I close the account and open a new one? 

        Thanks,
        Mary
        """,
     """
     Dear Mary, 

    Service address: Room 1001, 10/F ABC Building, 100 Oil Street, North Point, HK

    Thank you for your email.

    For moving house, the registered customer has to arrange termination of the current electricity account and set up a new one for the new address. Would you please advise the registered customer to contact our Customer Services Executives at 2887 3411 during office hours or complete the electronic forms at the path below to process the requests: 

    Account Termination (with at least 2 working days’ advance notice and should not fall on Saturday / Sunday / Public Holiday)
    https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US

    For refund of the deposit balance (after deducting the final outstanding as at the termination date from the deposit), we could send a crossed cheque in Hong Kong Dollars payable to the registered customer name to the new correspondence address in around 5 working days upon closure of account. 

    Account Registration (with at least 1 working day advance notice and should not fall on Saturday / Sunday / Public Holiday)
    https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US

    Upon successful registration, a deposit is required as security for future use of electricity and a new electricity account number will be assigned. The electricity supply will be connected during office hours on the account effective date. 

    If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours. 
    Yours sincerely,
    YM Lai
    Senior Manager (Customer Services)
    HK Electric
     """),
    QAPair("""
    	Hello Customer Service Team, 

    	I'm moving out of my current apartment very soon and need to cancel my electricity service. My existing account number is 1234567890.  Can you please let me know how I can unsubscribe from my current electricity account? Do I need to provide any documentation or give advance notice?

    	Thank you!
    	Mary 
    	""",
     """
     Dear Mary,

     Account no: 1234567890

     Thank you for your email below.

     To arrange account termination and deposit refund, would you please complete the electronic form at the path below to process the request:

     Account Termination (with at least 2 working days’ advance notice and should not fall on Saturday / Sunday / Public Holiday)
     https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US

     For refund of the deposit balance (after deducting the final outstanding as at the termination date from the deposit), we could send a crossed cheque in Hong Kong Dollars payable to the registered customer name to the new correspondence address in around 5 working days upon closure of account. 

     If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours.
     Yours sincerely,
     YM Lai
     Senior Manager (Customer Services)
     HK Electric
     """),
    QAPair("""
    	Dear colleague,

    	I would like to close my current account 1234567890 as my new tenant will open a new account.

    	Please advise the necessary procedure.

    	Thank you,

    	Mary
    	""",
     """
     Dear Mary,

     Account no: 1234567890

     Thank you for your email below.

     To arrange account termination and deposit refund, would you please complete the electronic form at the path below to process the request:

     Account Termination (with at least 2 working days’ advance notice and should not fall on Saturday / Sunday / Public Holiday)
     https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US

     For refund of the deposit balance (after deducting the final outstanding as at the termination date from the deposit), we could send a crossed cheque in Hong Kong Dollars payable to the registered customer name to the new correspondence address in around 5 working days upon closure of account. 

     If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours.
     Yours sincerely,
     YM Lai
     Senior Manager (Customer Services)
     HK Electric
     """),
    QAPair("""
    	Dear Customer Service Team, 

    	I am writing to express interest in receiving my monthly electricity bill by email or accessing it online through your website. Can you please let me know how I can set this up, and what steps I need to take to ensure that I receive my bills electronically? Do I need to provide any documentation or create an online account? My existing account number is 0987654321.

    	I'm trying to reduce paper waste and simplify my bill-paying process, so any information you can provide would be helpful. 

    	Thank and regards,

    	Mary
    	""",
     """
     Dear Mary, 

     Account no: 0987654321

     Thank you for your email below. 

     If you would like to receive e-bills instead of hardcopy bills, please register for the “Account-On-Line” and set up e-bill service via one of the following channels:

     1.	Access the AOL Fast Track Registration by scanning the "E-bill Registration" QR code printed aside the right of the customer address on the top of recent electricity bill copy;

     2.	Complete the online application via https://aol.hkelectric.com/AOL/aol#/account/preregistration. After the online registration, a confirmation letter with an activation code will be sent to the correspondence address of the account by post. Please follow the instructions in the letter to complete the one-off activation process and start receiving e-bills. Alternatively, if you would like HK Electric to activate the service for you, please provide your HKID card / passport copy for our verification by replying to this email.

     You may also login the AOL service for accessing the account information after registration.

     If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours. 
     Yours sincerely,
     YM Lai
     Senior Manager (Customer Services)
     HK Electric
     """),
    QAPair("""
    	Dear CS,

    	Our account no is 0987654321.
    	Can you tell me what is our current account balance? And how can I download softcopy bill and check the account balance myself? please advise, thank you!　

    	Thanks & Regards,
    	Mary

    	""",
     """
     Dear Mary, 

     Account no: 0987654321

     Thank you for your email below. 

     If you would like to receive e-bills instead of hardcopy bills, please register for the “Account-On-Line” and set up e-bill service via one of the following channels:

     1.	Access the AOL Fast Track Registration by scanning the "E-bill Registration" QR code printed aside the right of the customer address on the top of recent electricity bill copy;

     2.	Complete the online application via https://aol.hkelectric.com/AOL/aol#/account/preregistration. After the online registration, a confirmation letter with an activation code will be sent to the correspondence address of the account by post. Please follow the instructions in the letter to complete the one-off activation process and start receiving e-bills. Alternatively, if you would like HK Electric to activate the service for you, please provide your HKID card / passport copy for our verification by replying to this email.

     You may also login the AOL service for accessing the account information after registration.

     If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours. 
     Yours sincerely,
     YM Lai
     Senior Manager (Customer Services)
     HK Electric
     """),
    QAPair("""
    	Hi, I recently received my electricity bill and I'm wondering what payment methods are available. My existing account number is 0123123123.  Can you please let me know how I can pay my bill, and what options I have for payment? Do you accept credit cards, or bank transfers?  Any information you can provide would be much appreciated. 
    	Thank you.
    	Mary
    	""",
     """
     Dear Mary,

     Account no: 0123123123

     Thank you for your email below.

     As discussed, you may refer to the link below of our corporate website for various payment methods: 
     https://www.hkelectric.com/en/customer-services/billing-payment-electricity-tariffs/how-to-pay-bill

     If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours.

     Yours sincerely,

     YM Lai
     Senior Manager (Customer Services)
     HK Electric
     """),
    QAPair("""
    	Dear Sir/Madam,

    	Can you provide the payment methods for me to settle the electricity bill? Thank you!

    	Regards,
    	Mary
    	""",
     """
     Dear Mary,

     Account no: 0123123123

     Thank you for your email below.

     As discussed, you may refer to the link below of our corporate website for various payment methods: 
     https://www.hkelectric.com/en/customer-services/billing-payment-electricity-tariffs/how-to-pay-bill

     If you have further enquiries, please contact our Customer Services Executives at 2887 3411 during office hours.

     Yours sincerely,

     YM Lai
     Senior Manager (Customer Services)
     HK Electric
     """),
    QAPair("""
    	Dear HK Electric, 

    	I'm planning to relocate to a new address: Room 908, 9/F WTT Tower, Yuen Long, New Territories, Hong Kong next month.  Let me know what information or documentation I'll need to provide? Thank you

    	Regards,
    	Mary
    	""",
     """
     Dear Mary,
     Service address: Room 908, 9/F World Trade Tower, Yuen Long, New Territories, Hong Kong
     Thank you for using our electronic form.

     Please be informed that our Company is responsible for electricity supply for Hong Kong Island and Lamma Island. As your service address is in New Territories, please contact China Light and Power at csd@clp.com.hk or 2678 2678 for your request.
     Yours faithfully,
     YM Lai
     Senior Manager (Customer 
     """),
    QAPair("""
    	Hello,

    	I would like to set up a new account in Yuen Long.

    	Address is Room 908, 9/F WTT Tower, Yuen Long, New Territories, Hong Kong

    	I will move on 26th June 2023.

    	Please let me know what are the next steps,

    	Thanks,
    	Mary
    	""",
     """
     Dear Mary,
     Service address: Room 908, 9/F World Trade Tower, Yuen Long, New Territories, Hong Kong
     Thank you for using our electronic form.

     Please be informed that our Company is responsible for electricity supply for Hong Kong Island and Lamma Island. As your service address is in New Territories, please contact China Light and Power at csd@clp.com.hk or 2678 2678 for your request.
     Yours faithfully,
     YM Lai
     Senior Manager (Customer 
     """)
]

# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('./data/', glob='**/*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()
logger.info(f'documents:{len(documents)}')

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# LLM Model
llm = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key=zhipuai.api_key)

# Prompt-Template

custom_question_prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=RAG_TEMPLATE)

# RAG VectorSearch: 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
vector_search = Chroma.from_documents(split_docs, embedding=GPT4AllEmbeddings())

# 定义内存记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
for qa in samples:
    memory.chat_memory.add_user_message(re.sub(' +', ' ', qa.question))
    memory.chat_memory.add_ai_message(re.sub(' +', ' ', qa.answer))

def querying(query, history):

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_search.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_question_prompt}
    )
    logger.info("memory:")
    logger.info(memory.chat_memory.messages)
    logger.info("question: " + INSTRUCTION + query)

    result = qa_chain({"question": INSTRUCTION + query})
    logger.info("answer: " + result["answer"].strip())

    return result["answer"].strip().replace("\\n", "</br>")


# Launch the interface
#gr.ChatInterface(querying).launch(share=False)
gr.ChatInterface(querying).launch(share=True)
