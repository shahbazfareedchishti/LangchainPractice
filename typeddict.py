from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import TypedDict

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0.7
)

 
prompt = """
    The Peaky Blinders film is a cold, uncompromising funeral for a man who died spiritually a decade ago, and while you may hate that Tommy Shelby finally met his end, any other conclusion would have been narrative cowardice. To keep him alive would be to ignore the absolute truth of his trajectory: Tommy was a structural anomaly fueled by trauma and ambition, a ghost haunting the living world until his ledger was finally written in blood. The movie leans into this inevitability with a bleak, cinematic grandeur, stripping away the romanticism of the Birmingham underworld to reveal the hollow core of the Shelby empire. Cillian Murphy portrays this collapse with a weary brilliance, showing us a man who has run out of soldier's minutes. You are grieving a character who was never meant to survive his own sins, and while his death feels like a betrayal of your investment, it is actually the ultimate respect to his legacy, giving him the one thing he could never achieve in life: peace.
"""

structured_llm = llm.with_structured_output(MyClass)
response = structured_llm.invoke(prompt)
print(response)