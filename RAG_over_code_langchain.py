#before launchaing this script, make sure you have an instance of an LLM loaded accessible by the enpoint :3000
#example of Source code analysis with Langchain.

from langchain_community.llms import OpenLLM
import torch
from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline

torch.FloatTensor(0).to('cuda') #To change depending of the GPU you are using
server_url = "http://localhost:3000"
llm = OpenLLM(server_url=server_url)


# Clone
from datasets import load_dataset

#repo_path = "/home/infres/yokoyama/test_DebugBench" #folder corresponding of the source code for the RAG
#repo = Repo.clone_from("https://github.com/esphome/esphome", to_path=repo_path) #downloading the source code for the RAG
# Specify the dataset name and the column containing the content
dataset_name = "Rtian/DebugBench"

page_content_column = ("buggy_code" ,"solution", "solution_explanation")  # or any other column you're interested in

# Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# Load the data
data = loader.load()

# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)


# Load all python project files
"""loader = GenericLoader.from_filesystem(
    repo_path + "",
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)


documents = loader.load()"""


"""from langchain.text_splitter import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
"""
#Split the Document into chunks for embedding and vector storage.
#texts = python_splitter.split_documents(documents)

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)


# Prompt
template = """Use the context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context","problem", "question"],
    template=template,
)

# Docs
context = "class Solution: def longestValidSubstring(self, word: str, forbidden: List[str]) -> int: trie = {} for f in forbidden: t = trie for c in f: if c not in t: t[c] = {} t = t[c] t['end'] = True def isForbidden(s): t = trie counter = 0 for c in s: if c not in t: return False t = t[c] counter += 1 if 'end' in t: return counter return False res = 0 j = len(word) + 1 for i in range(len(word) - 1, -1, -1): truc = isForbidden(word[i:j]) if truc: j = i + truc - 1 res = max(res, j - i) return res"
question = "The code in the context is supposed to answer the following problem, but there is a small bug. You are given a string word and an array of strings forbidden. A string is called valid if none of its substrings are present in forbidden. Return the length of the longest valid substring of the string word. A substring is a contiguous sequence of characters in a string, possibly empty. Then, find and explain why there is a bug. Then, modify one line in the code in the context that solves this issue."
docs = retriever.get_relevant_documents(question)
# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

output = chain.invoke({"input_documents": docs, "question": question}, return_only_outputs=True)

print(output['output_text'])
