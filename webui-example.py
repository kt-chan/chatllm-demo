import os
import random
import gradio as gr

def random_response(message, history):
    return random.choice(["Yes", "No"])

demo = gr.ChatInterface(random_response)

os.environ["http_proxy"] = "http://c00627809:%40hwKT1986c@proxyhk.huawei.com:8080"
os.environ["https_proxy"] = "http://c00627809:%40hwKT1986c@proxyhk.huawei.com:8080"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
port=8081

#demo.launch(share=True, server_port=port)
demo.launch(server_port=port)