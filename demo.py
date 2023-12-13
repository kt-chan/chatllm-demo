
import os, logging
import zhipuai
import gradio as gr

# Import langchain stuff
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from llms.zhipuai_llm import ZhipuAILLM

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # 读取并加载环境变量，来自 .env 文件

os.environ["http_proxy"] = os.environ["PROXY"]
os.environ["https_proxy"] = os.environ["PROXY"]
os.environ["no_proxy"] = os.environ["NO_PROXY"]

refmeta = """
        Please follow the below instruciton in generating your output:

        Use electronic application whenever appropriate.
        For account openning and account transfer use this link: https://aol.hkelectric.com/AOL/aol#/eforms/appl?lang=en-US; and for account closing use this link: https://aol.hkelectric.com/AOL/aol#/eforms/term?lang=en-US;
        For account termination or account close, always provide information related to deposit refund and it is preferred to use crossed cheque made payable if applicable.
        For relocation, always first provide information about account termination, and then account openning.
        And You have to restrict your answer with the information provided in this ref tag, and only provide the best option and do not list all available options. 

        
        Move In
        Welcome to "Electricity Account Management". This Section contains 5 modules. Read " Move In " if you are going to move house. Don't skip " Billing & Payment" if you wish to manage your account in a more effective way. Also take time to see " Use of Electricity " for more information on electrical safety and energy efficiency. If you would vacate your premises, " Move Out " and "Deposit Refund " would be most useful to you. 

        Please click the description that suits your new premises:

        1.
        Electricity supply is available at my new premises and there is no alteration made to the existing installation and / or the supply loading or ​Electricity supply at my new premises has been disconnected for not more than four months and there is no alteration made to the existing installation and / or the supply loading.
        faq close
        Please arrange transfer of account of the new address to make it under your name.

        In most of the cases, you can arrange the transfer of account by

        Complete and submit the respective application form via HK Electric App.
        Completing the respective electronic application form on the Website.
        Calling our Customer Services Executives at 2887 3411 during office hours.
        Submitting an application form to our Customer Centre or by fax to 2510 7667.
        Notice in Advance

        Please give one working day advance notice.

        What will Hongkong Electric do on the transfer effective date?

        We will take a meter reading and connect electricity supply (if electricity supply is not available). It is not necessary to make appointment with us for taking meter reading unless the electricity meter is inside the premises.

        How about deposit?

        A deposit is required as security for future use of electricity. The required deposit is equivalent to 60 days estimated consumption, and the estimation is based on the loading of appliances and the main switch rating. The deposit will be refunded to the registered customer upon termination of account.


        2.
        There is alteration made to the existing installation and / or the supply loading at my new premises or Electricity supply at my new premises has been disconnected for more than four months.
        faq close
        Please apply for new supply. We will inspect the installation at the customer's premises before connection of supply.

        In most of the cases, you can apply for new supply by:

        Completing the respective electronic application form on the Website.
        Submitting an application form to our Customer Centre or by fax to 2510 7667.
        Calling our Customer Services Executives at 2887 3411 during office hours.
        Inspection on the Installation of the Customer's Premises

        You may make appointment for installation inspection with us if you have submitted the application for new supply.
        The registered electrical contractor / worker should submit a copy of the duly completed "Work Completion Certificate (WCC)" on or before the installation inspection and the registered electrical worker of the appropriate grade should be present on site during the installation inspection.
        Connection of Supply

        Normally, upon satisfactory installation inspection, electricity supply will be connected immediately. If the result is unsatisfactory, re-inspection is required and re-inspection fees will be levied.
        If the application for supply requires extra equipment and / or application for official permits, it may take a longer time and service charge may be required.
        If an installation is connected to communal rising mains and its main switch rating has to be increased, "CI Form 140" should be submitted to confirm that it is agreed by the owner of rising mains.
        How about deposit?

        A deposit is required as security for future use of electricity. The required deposit is equivalent to 60 days estimated consumption, and the estimation is based on the loading of appliances and the main switch rating. The deposit will be refunded to the registered customer upon termination of account.

        Move Out
        Welcome to "Electricity Account Management". This Section contains 5 modules. Read " Move In " if you are going to move house. Don't skip " Billing & Payment " if you wish to manage your account in a more effective way. Also take time to see " Use of Electricity " for more information on electrical safety and energy efficiency. If you would vacate your premises, " Move Out " and "Deposit Refund " would be most useful to you. 

           

        Are you the registered customer of the electricity account of the premises?
        1.
        Yes, I am the registered customer of the electricity account of the premises.
        faq close
        Please arrange termination of account under your name if you are going to move out from the premises.

        Please notify us two working days in advance. You may arrange termination of account via one of the following channels:

        Complete the respective electronic application form on our website.
        Call our Customer Services Executives at 2887 3411 during office hours.
        Complete and return the form "Application for Termination of Electricity Account" to our Customer Centre or by fax to 2510 7667.
        It is not necessary to make appointment with us for taking the final meter reading unless the electricity meter is inside the premises. 
        If there is a new occupant, the account will be automatically finalized on the effective transfer date of the application for transfer from the new customer. However, the registered customer is liable for all outstanding charges of the account as long as the account remains under his name.


        2.
        No, I am not the registered customer of the electricity account of the premises. The registered customer is my landlord / the ex-landlord / the ex-occupant.
        faq close
        You may ask for a special meter reading on the date that you will vacate the premises with one working day's notice in advance. The special meter reading can be arranged via one of the following channels:

        Complete the respective electronic form on our website.
        Call our Customer Services Executives at 2887 3411 during office hours.
        It is not necessary to make appointment with us for taking special meter reading unless the electricity meter is inside the premises.
        
        Deposit Refund
        Welcome to "Electricity Account Management". This Section contains 5 modules. Read " Move In " if you are going to move house. Don't skip " Billing & Payment " if you wish to manage your account in a more effective way. Also take time to see " Use of Electricity " for more information on electrical safety and energy efficiency. If you would vacate your premises, " Move Out " and "Deposit Refund " would be most useful to you. 

           

        For registered customers issued with deposit receipt
        By Cheque

        Please mail the properly-endorsed deposit receipt, together with the correspondence address and telephone no. to our Customer Centre. A crossed cheque made payable to the registered customer will then be mailed to the correspondence address within five working days.

        Direct Refund to Bank Account

        For the refundable amount $5,000 or below, we can also arrange direct refund to the bank account of the registered customer in Hong Kong within five working days upon receipt of a copy of bank record showing the bank account no. and bank account name.

        If deposit receipt is lost, please call us at 2887 3411 during office hours. A letter of indemnity will be sent to the registered customer for completion.

        For registered customers not issued with a deposit receipt (i.e. The deposit was paid with the first electricity bill of your account)
        Please call our Customer Services Executives at 2887 3411 during office hours to arrange deposit refund.

        A crossed cheque made payable to the registered customer will be mailed to the correspondence address within five working days.
        If the refundable amount is at $5,000 or below, we can also arrange direct refund to the bank account of the registered customer in Hong Kong within five working days upon receipt of a copy of bank record showing the bank account no. and bank account name.
        Deposit of the electricity account can be refunded upon termination of account.
"""

zhipuai.api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息

llm = ZhipuAILLM(model="chatglm_turbo", temperature=0.9, top_p=0.1, zhipuai_api_key= zhipuai.api_key)

def output_parser(message):
    return message.replace('\\n', '</br>')

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    message=message+" "+refmeta
    logging.warning("input:" + message)
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm.predict_messages(history_langchain_format)
    response = output_parser(gpt_response.content)
    logging.warning("output:" + response)
    return response

# Launch the interface
gr.ChatInterface(predict).launch()

#print(llm.generate(['什么llm封装']))