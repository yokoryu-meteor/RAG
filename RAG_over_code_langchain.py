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
from langchain_community.embeddings import HuggingFaceEmbeddings

torch.FloatTensor(0).to('cuda') #To change depending of the GPU you are using
server_url = "http://localhost:3000"
llm = OpenLLM(server_url=server_url)


# Clone
repo_path = "/home/infres/mcaillard-23/test_starcoder" #folder corresponding of the source code for the RAG
#repo = Repo.clone_from("https://github.com/esphome/esphome", to_path=repo_path) #downloading the source code for the RAG


# Load all python project files
loader = GenericLoader.from_filesystem(
    repo_path + "",
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)


documents = loader.load()


from langchain.text_splitter import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)

#Split the Document into chunks for embedding and vector storage.
texts = python_splitter.split_documents(documents)

# embed and save in vector_store
sentence_t5_base = "sentence-transformers/sentence-t5-base"
embeddings = HuggingFaceEmbeddings(
    model_name=sentence_t5_base,
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"device": "cuda"},
            )
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)


# Prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Docs
context="class Solution { public: void bfs(int prevcolor,int row,int col,int color,vector<vector<int>>&visited,vector<vector<int>>&image) { // it'll ease the work queue<pair<int,int>> q; q.push({row,col}); visited[row][col]=1; while(!q.empty()) { row = q.front().first; col = q.front().second; q.pop(); // if img is of the starting color. if(image[row][col]==prevcolor) image[row][col]=color; // up if(row-1>=0 && !visited[row-1][col] && image[row-1][col]==prevcolor) q.push({row-1,col}),visited[row-1][col]=1, image[row-1][col]=color; // right if(col+1<image[0].size() && !visited[row][col+1] && image[row][col+1]==prevcolor) q.push({row,col+1}),visited[row][col+1]=1, image[row][col+1]=color; //down if(row+1<image.size() && !visited[row+1][col] && image[row+1][col]==prevcolor) q.push({row+1,col}),visited[row+1][col]=1, image[row+1][col]=color; // left if(col-1>=0 && !visited[row][col-1] && image[row][col-1]==prevcolor) q.push({row,col-1}),visited[row][col-1]=1, image[row][col-1]=color; } } vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) { int prevcolor = image[sr][sc]; // visiting array. vector<vector<int>> visited(image.size(),vector<int>(image[0].size(),0)); bfs(prevcolor,sr,sc,color,visited,image); return image; } };"
# Docs
question = "find the bugs and explain them in the code in the context. An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image. You are also given three integers sr, sc, and color. You should perform a flood fill on the image starting from the pixel image[sr][sc]. To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with color. Return the modified image after performing the flood fill, with the following constraints: (m == image.length n == image[i].length 1 <= m, n <= 50 0 <= image[i][j], color < 216 0 <= sr < m 0 <= sc < n)"
docs = retriever.get_relevant_documents(question)
# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

output = chain.invoke({"input_documents": docs, "question": question}, return_only_outputs=True)

print(output['output_text'])
