from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# Use MessagesPlaceholder to manage history cleanly
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that writes blogs about {topic}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{topic}")
])

chat_history = []

while True:
    user_input = input("\nIdeas or Instruction (Type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Thank you for using blog generator")
        break

    # 1. Format the messages using the template and the history list
    chain_input = {
        "topic": user_input,
        "history": chat_history
    }
    
    # 2. Invoke the LLM
    # We use the prompt to format everything into the correct structure
    formatted_prompt = prompt.format_prompt(**chain_input)
    response = llm.invoke(formatted_prompt.to_messages())
    
    print("\n--- Blog Output ---\n")
    print(response.content)
    
    # 3. Update History (Crucial for it to actually work)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))