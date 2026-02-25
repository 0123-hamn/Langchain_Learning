# model 1
from langchain_ollama import OllamaLLM # model 2
from langchain_groq import ChatGroq # model 3
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence


load_dotenv()


model2 = OllamaLLM(model='tinyllama')

model3 = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template = 'Generate a 2 para notes of text \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate  5 question and answer from text \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz in single document \n notes -> {notes} and quiz ->{quiz}',
    input_variables = ['notes','quiz']
)

parser = StrOutputParser()

parallel_chain =  RunnableParallel({
    'notes': RunnableSequence(prompt1,model2,parser),
    'quiz': RunnableSequence(prompt2,model3,parser)
})
merge_chain = RunnableSequence(prompt3,model2,parser)
text = '''
random forests (see RandomForestClassifier and RandomForestRegressor classes), each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set.

During the construction of each tree in the forest, a random subset of the features is considered. The size of this subset is controlled by the max_features parameter; it may include either all input features or a random subset of them (see the parameter tuning guidelines for more details).

The purpose of these two sources of randomness (bootstrapping the samples and randomly selecting features at each split) is to decrease the variance of the forest estimator. Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking an average of those predictions, some errors can cancel out. Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias. In practice the variance reduction is often significant hence yielding an overall better model.

When growing each tree in the forest, the “best” split (i.e. equivalent to passing splitter="best" to the underlying decision trees) is chosen according to the impurity criterion. See the CART mathematical formulation for more details.

In contrast to the original publication [B2001], the scikit-learn implementation combines classifiers by averaging their probabilistic prediction, instead of letting each classifier vote for a single class.

A competitive alternative to random forests are Histogram-Based Gradient Boosting (HGBT) models:

Building trees: Random forests typically rely on deep trees (that overfit individually) which uses much computational resources, as they require several splittings and evaluations of candidate splits. Boosting models build shallow trees (that underfit individually) which are faster to fit and predict.

Sequential boosting: In HGBT, the decision trees are built sequentially, where each tree is trained to correct the errors made by the previous ones. This allows them to iteratively improve the model’s performance using relatively few trees. In contrast, random forests use a majority vote to predict the outcome, which can require a larger number of trees to achieve the same level of accuracy.

Efficient binning: HGBT uses an efficient binning algorithm that can handle large datasets with a high number of features. The binning algorithm can pre-process the data to speed up the subsequent tree construction (see Why it’s faster). In contrast, the scikit-learn implementation of random forests does not use binning and relies on exact splitting, which can be computationally expensive.

Overall, the computational cost of HGBT versus RF depends on the specific characteristics of the dataset and the modeling task. It’s a good idea to try both models and compare their performance and computational efficiency on your specific problem to determine which model is the best fit.
'''

chain = RunnableSequence(parallel_chain,merge_chain)
result = chain.invoke({'text':text})

print(result)