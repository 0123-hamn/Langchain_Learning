from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a report on following {topics}',
    input_variables = ['topics'] 
)

prompt2 = PromptTemplate(
    template = 'Generate a 5 points summary of the {text}',
    input_variables = ['text']
)

model = OllamaLLM(model = "tinyllama")
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topics':'Agentic AI'})
print(result)