
import os
import zhipuai
import gradio as gr

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # 读取并加载环境变量，来自 .env 文件

# Import langchain stuff
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from zhipuai_llm import ZhipuAILLM

os.environ['OPENAI_API_KEY'] = os.environ["ZHIPUAI_API_KEY"]
zhipuai.api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息
llm = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key= zhipuai.api_key)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm.predict_messages(history_langchain_format)
    return gpt_response.content

# Launch the interface
gr.ChatInterface(predict).launch()

#print(llm.generate(['什么llm封装']))