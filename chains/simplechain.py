from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = 'Generate 5 interesting points about {topics}',
    input_variable = ['topics']
)


model = OllamaLLM(model = "tinyllama")

parser = StrOutputParser()

chain = prompt|model|parser

result = chain.invoke({'topics':'Cricket'})

print(result)

chain.get_graph().print_ascii()