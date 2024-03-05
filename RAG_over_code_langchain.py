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
context="class Solution: def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]: tree = None ans = [] for i, x in enumerate(nums): tree = insert(tree, x) if size(tree) > k: tree = remove(tree, nums[i - k + 1]) if size(tree) == k: if k % 2 == 1: ans.append(get(tree, k // 2 + 1)) else: nums.append((get(tree, k // 2) + get(tree, k // 2 + 1)) / 2) return nums class Node: __slots__ = ['val', 'count', 'weight', 'size', 'left', 'right'] def __init__(self, val): self.val = val self.count = 1 self.weight = random.random() self.size = 1 self.left = self.right = None def touch(root): if not root: return root.size = root.count + size(root.left) + size(root.right) def size(root): if not root: return 0 return root.size def insert(root, val): t1, r, t2 = split(root, val) if not r: r = Node(val) else: r.count += 1 touch(r) t2 = join(r, t2) return join(t1, t2) def remove(root, val): t1, r, t2 = split(root, val) if r and r.count > 1: r.count -= 1 touch(r) t2 = join(r, t2) return join(t1, t2) def split(root, val): if not root: return None, None, None elif root.val < val: a, b, c = split(root.right, val) root.right = a touch(root) return root, b, c elif root.val > val: a, b, c = split(root.left, val) root.left = c touch(root) return a, b, root else: a, c = root.left, root.right root.left = root.right = None touch(root) return a, root, c def join(t1, t2): if not t1: return t2 elif not t2: return t1 elif t1.weight < t2.weight: t1.right = join(t1.right, t2) touch(t1) return t1 else: t2.left = join(t1, t2.left) touch(t2) return t2 def get(root, index): if size(root.left) < index <= size(root.left) + root.count: return root.val elif size(root.left) + root.count < index: return get(root.right, index - root.count - size(root.left)) else: return get(root.left, index)"
question = "find the bugs and explain them in the code in the context. The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle values. For examples, if arr = [2,3,4], the median is 3. For examples, if arr = [1,2,3,4], the median is (2 + 3) / 2 = 2.5. You are given an integer array nums and an integer k. There is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the median array for each window in the original array. Answers within 10-5 of the actual value will be accepted. here are the constraints: 1 <= k <= nums.length <= 105 -231 <= nums[i] <= 231 - 1"
docs = retriever.get_relevant_documents(question)
# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

output = chain.invoke({"input_documents": docs, "question": question}, return_only_outputs=True)

print(output['output_text'])
