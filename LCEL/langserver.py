from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Model Create korbo :
model = ChatGroq(model="llama-3.1-8b-instant")

# prompt template banabo 
system_template = "Translate the following into {langauage}: "
prompt_template = ChatPromptTemplate.from_messages([
    ('system',system_template),
    ('user','{text}')
])

# Parser banabo
parser = StrOutputParser()

# chain banabo = prompt_template|model|parser

chain = prompt_template|model|parser

## Building API by Langserve

app = FastAPI(title="Langserve_Learning_App")

add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

