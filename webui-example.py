import os
import random
import gradio as gr

def random_response(message, history):
    return random.choice(["Yes", "No"])

demo = gr.ChatInterface(random_response)
os.environ['OPENAI_API_KEY'] = os.environ["ZHIPUAI_API_KEY"]
os.environ["http_proxy"] = os.environ["PROXY"]
os.environ["https_proxy"] = os.environ["PROXY"]
os.environ["no_proxy"] = os.environ["NO_PROXY"]
port=8081

#demo.launch(share=True, server_port=port)
demo.launch(server_port=port)