from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

prompt1 = PromptTemplate(
    template ='generate a joke on the following :{topic}',
    input_variables = ['topic']
)

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'Explain the joke genrated : {text}',
    input_variable = ['text']
)

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))