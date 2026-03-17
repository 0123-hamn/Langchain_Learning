import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import  streamlit as st
import requests
import json
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/b5e4d85b0b3ecba055cb7518/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

llm_with_tools = llm.bind_tools([get_conversion_factor,convert])
st.title("Currency Converter AI")
user_input = st.text_input("Ask something like: Convert 10 USD to INR")
if user_input:
    messages = [HumanMessage(content=user_input)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        if tool_call["name"] == "get_conversion_factor":
            tool_message1 = get_conversion_factor.invoke(tool_call["args"])
            conversion_rate = tool_message1["conversion_rate"]
            messages.append(tool_message1)
        if tool_call["name"] == "convert":
            tool_call["args"]["conversion_rate"] = conversion_rate
            tool_message2 = convert.invoke(tool_call["args"])
            messages.append(tool_message2)
    llm_with_tools.invoke(messages).content


