from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content = "Langchain helps developers build LLM applications easily"),
    Document(page_content = "Chroma is a vector database optimized for LLM based search"),
    Document(page_content = "Embeddings convert text into high-dimensional vectors"),
    Document(page_content = "OpenAI provides powerful embedding models")
]

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents = documents,
    embedding = embeddings,
    collection_name = "my_collection"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3, "lambda_mult": 0.80}
)

query = "What is Chroma used for?"

results = retriever.invoke(query)

print(results)