import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

model1 = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0.7
)

model2 = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0.7
)

prompt1 = PromptTemplate(
    template = "Write a short and simple note from following {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 short questions from this {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""Merge the following notes and Q&A into a single structured summary document. 
Do not use a letter or email format. Please use markdown headings.

Notes:
{notes}

Q&A:
{qa}
""",
    input_variables=["notes", "qa"]
)

parser = StrOutputParser()

runnable_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'qa': prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

final_chain = runnable_chain | merge_chain


text = """

Constantine (820s or 830s – before 836) was an infant prince of the Amorian dynasty who briefly ruled as co-emperor of the Byzantine Empire sometime in the 830s, alongside his father Theophilos. He was born to Theophilos and his wife, Empress Theodora. When naming Constantine, his father defied standard naming conventions, as his son was not named after his father Michael II. Most information about Constantine's short life and titular reign is unclear, although it is known that he was born sometime in the 820s or 830s and was installed as co-emperor soon after his birth. He appears on the coinage issued under his father, albeit addressed as despot (not a formal title, but an honorific interchangeable with basileus, i.e. emperor) on gold coins, but with no title at all on bronze ones. He died sometime before 836, possibly after falling into a palace cistern. His parents buried him in a sarcophagus made of Thessalian marble in the Church of the Holy Apostles. 
"""

result = final_chain.invoke({"topic": text, "text": text})
print(result)
