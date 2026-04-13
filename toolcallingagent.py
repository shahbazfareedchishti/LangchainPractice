from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

llm=ChatGroq(
    model = "llama-3.3-70b-versatile",
)

@tool
def multiply(a:int, b:int)->int:
    """Multiply two integers together."""
    return a*b

@tool
def add(a:int, b:int)->int:
    """Add two integers together."""
    return a+b

tools = [multiply, add]

system_prompt = """You are a helpful AI assistant, use the tools when necessary to answer the questions, otherwise answer directly."""

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = system_prompt
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content": user_input}]
        }
    )
    print("AI: ", response["messages"][-1].content)
