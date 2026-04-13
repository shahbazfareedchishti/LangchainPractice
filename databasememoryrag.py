import streamlit as st
import sqlite3
import uuid
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

conn = sqlite3.connect("chat_history.db", check_same_thread = False)

cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        session_id TEXT,
        role TEXT,
        content TEXT
    )
""")

conn.commit()

def save_message(session_id: str, role: str, content: str):
    cursor.execute("""
        INSERT INTO chat_history (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))
    conn.commit()

def load_chat_history(session_id: str) -> List[BaseMessage]:
    cursor.execute("""
        SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY rowid
    """, (session_id,))
    rows = cursor.fetchall()
    history: List[BaseMessage] = []
    for role,content in rows:
        if role == "human":
            history.append(HumanMessage(content=content))
        elif role == "ai":
            history.append(AIMessage(content=content))
    return history

def get_all_sessions():
    cursor.execute("SELECT DISTINCT session_id FROM chat_history order by rowid desc")
    rows = cursor.fetchall()
    return [row[0] for row in rows]

st.set_page_config(
    page_title = "Database Memory RAG",
    layout = "wide"
)

st.title("Database Memory RAG")

st.sidebar.title("Chat History")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

if st.sidebar.button("New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("Previous Chats")

for sid in get_all_sessions():
    if st.sidebar.button(sid[:8]):
        st.session_state.session_id = sid
        st.session_state.chat_history = load_chat_history(sid)
        st.rerun()

session_id = st.session_state.session_id

@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("text.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = "my_collection"
    )
    return vectorstore

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs = {"k": 4}
)

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
    temperature = 0.7
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content = """You are a helpful AI assistant, answer the questions from given context"""),
    MessagesPlaceholder(variable_name = "chat_history"),
    (
        "human",
        "Context: \n {context} \n\n question: \n{input}"
    )
])

def conversation_rag(user_input: str, chat_history: List[BaseMessage]):
    docs = retriever.invoke(user_input)
    context = "\n\n".join(
        [f"[Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}" for d in docs]
    )
    message = prompt.invoke(
        {
            "input": user_input,
            "context": context,
            "chat_history": chat_history
        }
    )
    response=llm.invoke(message)
    return response, docs

if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history(session_id)

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("ai").write(msg.content)

user_input = st.chat_input("Ask a question")

if user_input:
    st.chat_message("user").write(user_input)
    save_message(session_id, "human", user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    response, sources = conversation_rag(user_input, st.session_state.chat_history)
    st.chat_message("ai").write(response.content)
    save_message(session_id, "ai", response.content)
    st.session_state.chat_history.append(AIMessage(content=response.content))