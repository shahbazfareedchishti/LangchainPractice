from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from judge import evaulate_answer
from prompt import QUESTION_PROMPT
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
)

def generate_answer(question: str)->str:
    prompt = QUESTION_PROMPT.format(question = question)
    response = llm.invoke(prompt)
    return response

def ask_until_good(question: str, threshold: int = 8, max_attempts: int = 3)->str:
    attempts = 0
    while attempts<max_attempts:
        answer = generate_answer(question)
        evaluation = evaulate_answer(question, answer)
        if evaluation.get("accuracy",0) >= threshold and not evaluation.get("hallucination",True):
            print("Answer is good")
            return answer
        else:
            print("Answer is not good, retrying...")
            attempts+=1
    return "Failed to generate a good answer"
    

question = "What is the capital of France?"
answer = ask_until_good(question)
print(answer)

        