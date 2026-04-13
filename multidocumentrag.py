from dotenv import load_dotenv

load_dotenv()

from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq
import os

DOC_FOLDER = "docs"

all_documents:List[Document] = []

for file in os.listdir(DOC_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOC_FOLDER, file))
        docs = loader.load()
        
        for d in docs:
            d.metadata["source"] = file        
        all_documents.extend(docs)

print(f"Loaded {len(all_documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_documents)
print(f"Total Chunks created: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

)

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    collection_name = "my_collection"
)

retriever = vectorstore.as_retriever(
    search_kwargs = {"k": 4}
)

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content = """You are a helpful AI assistant, answer the questions from given context, if the answer is not in the context then say I don't know"""
        ),
        (
            "human",
            "Context: \n {context} \n\n question: \n{input}"
        )
    ]
)

def multi_document_rag(query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join([
        f"""[Document: {d.metadata.get("source")}]\n
        {d.page_content}
        """ 
        for d in docs
    ])

    message = prompt.invoke(
        {
            "context" : context,
            "input" : query
        }
    )
    response = llm.invoke(message)
    return response.content,docs

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    answer, sources = multi_document_rag(query)
    print("AI: ",answer)
    print("\nSources: ")
    for i, d in enumerate(sources):
        print(f"{i+1}. {d.metadata['source']}")
    
    