from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

st.title('Research-paper ChatBot')

paper_input = st.text_input('Enter the name of research-paper')

length_input = st.selectbox('Select Explaination style',['1-2 para','2-5 para','5-7 para'])

style_input = st.selectbox('Select style',['Mathematical-based','code-based','theory-based'])

template = load_prompt('template.json')

if st.button('Summarize'):

    chain = template|model
    result = chain.invoke({
    "paper_input": paper_input,
    "length_input": length_input,
    "style_input": style_input
})


    st.write(result.content)