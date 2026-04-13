from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0.7
)

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "Write 5 lines on the following {text}",
    input_variables=["text"]
)

template3 = PromptTemplate(
    template = "Convert this into Urdu: {text}",
    input_variables=["text"]
)
output_parser = StrOutputParser()

chain = template1 | llm | output_parser | template2 | llm | output_parser | template3 | llm | output_parser

response = chain.invoke({'topic': "Black Holes"})
print(response)