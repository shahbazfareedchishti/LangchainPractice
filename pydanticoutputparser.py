from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, lt=60, description="Age of the person")
    city: str = Field(description="City of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template= (
        """
        Give me the name, age and city of a fictional {text} person
        Make sure the age is between 18 and 60
        Return the response in the following format
        {format_instructions}
        """
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

#prompt = template.invoke({'text': "Pakistan"})

chain = template | llm | parser

response = chain.invoke({'text': "Denmark"})
print(response)
    
