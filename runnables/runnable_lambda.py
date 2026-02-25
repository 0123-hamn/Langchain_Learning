from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_groq import ChatGroq

load_dotenv()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template = 'generate a joke on the following : {topic}',
    input_variables = ['topic']
)

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'cricket'})

final_result = """{}\n word count - {}""".format(result['joke'],result['word_count'])

print(final_result)