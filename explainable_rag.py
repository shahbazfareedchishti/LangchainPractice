from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("text.pdf")

documents = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = text_splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    collection_name = "my_collection"
)

query = "What is this document about?"

retrieved_docs=vectorstore.similarity_search(
    query = query,
    k = 3
)

context_text = ""
sources = []

for i, doc in enumerate(retrieved_docs):
    page=doc.metadata.get("page", "N/A")
    doc_source = doc.metadata.get("source", "PDF")
    
    context_text += f"\n Chunk {i+1}: \n{doc.page_content}\n"

    sources.append({
        "chunk": i+1,
        "page": page,
        "source": doc_source,
        "content": doc.page_content
    }) 


from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
)

prompt_template = PromptTemplate(
    template = """You are an AI assistant. Use the following context to answer the question. If the answer is not in the context, say so.
    Context: {context}
    Question: {question}
    
    """,
    input_variables = ["context", "question"]
)

parser = StrOutputParser()

chain = prompt_template | llm | parser

response = chain.invoke({
    "context": context_text,
    "question": query
})

print(response)

for src in sources:
    print(f"Chunk: {src['chunk']}) | Page: {src['page']}")
    print(src['content'])