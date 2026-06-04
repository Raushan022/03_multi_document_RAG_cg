from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()
SHOW_RETRIEVED_CHUNKS = False

# same embedding model used while creating FAISS   
embeddings = OpenAIEmbeddings(
   model="text-embedding-3-small"
)

# load existing FAISS index
vector_store = FAISS.load_local(
   "faiss_index",
   embeddings,
   allow_dangerous_deserialization=True
)

print("vector store loaded successfully")

# create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)

llm = ChatOpenAI(
   model="gpt-4o-mini",
    temperature=0
)

while True:
   question = input(
      "\nAsk a question (or type 'exit'): "
   )

   if question.lower() == "exit":
      print("Goodbye!")
      break

   # retrieve related chunks
   results = retriever.invoke(question)

   print(
    f"\nRetrieved {len(results)} chunks"
)

   if SHOW_RETRIEVED_CHUNKS:

      for index, doc in enumerate(results, start=1):
         print(f"\nChunk {index}")

         print(
            f"source: {doc.metadata.get('source')}"
         )

         print(
            f"Page: {doc.metadata.get('page')}"
         )

         print("\ncontent: ")

         print(
            doc.page_content[:500]
         )
         print(f"content length: {len(doc.page_content)}")

         print("\n" + "-" * 80)

   context = "\n\n".join(
      doc.page_content
      for doc in results
   )

   prompt = f"""
You are a helpful research assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""

   response = llm.invoke(prompt)

   print("\nAnswer:")
   print(response.content)

   print("=" * 80)

   # extracting sources
   sources = set()

   for doc in results:

      source = doc.metadata['source']
      page = doc.metadata["page"]

      sources.add(
         f"{source} (Page {page})"
      )

   print("\nSources:")

   for source in sorted(sources):
      print(source)

   print("\n" + "=" * 80)


# If the answer cannot be found in the context,
# say "I could not find that information in the documents."