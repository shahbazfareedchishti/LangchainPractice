from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

loader = PyPDFLoader("resume_shahbazfareed.pdf")

documents= loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
)

text = text_splitter.split_documents(documents)

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
    search_kwargs = {"k": 2, "lambda_mult": 0.80}
)

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
)

prompt_template = PromptTemplate(
    template="""You are an AI assistant. Use the following context to answer the question. If the answer is not in the context, say so."
    context: {context}
    question: {question}
""",
    input_variables = ["context", "question"]
)

output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    } | prompt_template | llm | output_parser
)

query = "Name skills mentioned in this document"

print(rag_chain.invoke(query))