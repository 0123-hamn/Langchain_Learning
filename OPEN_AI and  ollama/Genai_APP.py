import os
from dotenv import load_dotenv

import streamlit as st
from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Ollama-Chat")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond like a technical interview expert."),
        ("user", "Question: {question}")
    ]
)


st.title("Ollama-ChatBot")
input_text = st.chat_input("Ask your technical interview question:")


llm = OllamaLLM(model="tinyllama")


output_parser = StrOutputParser()


chain = prompt | llm | output_parser


if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
