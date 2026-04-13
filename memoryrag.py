from langchain_core.prompts import MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("text.pdf")
documents = loader.load()

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs = {"k": 4}
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content = "You are a helpful AI agent, answer the questions from given context"),
    MessagesPlaceholder(variable_name = "chat_history"),
    (
        "human",
        "Context: \n {context} \n\n question: \n{input}"
    )
])

def conversational_rag(user_input: str, chat_history: List[BaseMessage]):
    docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in docs)

    message = prompt.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
    })

    response = llm.invoke(message)
    return response, docs
    
chat_history: List[BaseMessage] = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    chat_history.append(HumanMessage(content = user_input))
    response, docs = conversational_rag(user_input, chat_history=chat_history)
    chat_history.append(AIMessage(content = response.content))
    print("AI: ", response.content)

