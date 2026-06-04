import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

all_docs = []

files = os.listdir("./data")

for file in files:
   if file.endswith(".pdf"):
      print(f"loading file {file}")

      full_path = "./data/" + file

      loader = PyPDFLoader(full_path)
      docs = loader.load()

      # print(len(docs))
      all_docs.extend(docs)

print("Total pages ", len(all_docs))

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size = 1000,
   chunk_overlap = 200
)

chunks = text_splitter.split_documents(all_docs)

embeddings = OpenAIEmbeddings(
   model="text-embedding-3-small"
)

vector_store = FAISS.from_documents(
   chunks,
   embeddings
)

vector_store.save_local("faiss_index")

print("Vector DB Created Successfully")
