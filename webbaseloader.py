from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7
)

prompt = PromptTemplate(
    template = "answer the following question {question} from the following text {text}",
    input_variables=["question", "text"]
)

url = "https://techlekh.com/dongfeng-nammi-vigo-price-nepal/"

loader =WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({"question": "What is this article about?", "text": docs[0].page_content})
print(response)
