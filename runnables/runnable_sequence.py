from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'Explain the joke generated : {text}',
    input_variables = ['text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))

