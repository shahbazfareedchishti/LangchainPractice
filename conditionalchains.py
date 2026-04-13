from langchain_core.output_parsers import format_instructions
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback, must be either positive or negative")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text into postive or negative: {text}""\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to the following feedback: {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to the following feedback: {text}",
    input_variables=["text"]
)

branch_chain = RunnableBranch(
    (lambda x: x["feedback"].sentiment == "positive", prompt2 | model | parser),
    (lambda x: x["feedback"].sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: x["feedback"].sentiment)
)

chain = {"feedback": classifier_chain, "text": lambda x: x["text"]} | branch_chain

response = chain.invoke({"text": "I hate this product"})
print(response)
    