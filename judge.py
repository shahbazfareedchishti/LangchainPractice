from langchain_groq import ChatGroq
from prompt import JUDGE_PROMPT
import json
from pydantic import BaseModel, Field

class Evaluation(BaseModel):
    accuracy: int = Field(..., ge=0, le=10)
    hallucination: bool = Field("True if the answer contains hallucnation else false")
    feedback: str = Field("brief comment on the quality of the answer")

judge_llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",    
)

structured_judge_llm = judge_llm.with_structured_output(Evaluation)

def evaulate_answer(question:str, answer:str)->dict:
    prompt = JUDGE_PROMPT.format(question = question, answer = answer)
    try:
        response = structured_judge_llm.invoke(prompt)
        evaluation = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    except Exception as e:
        evaluation={
            "accuracy": 0,
            "hallucination": True,
            "feedback": f"Failed to parse evaluation: {e}"
        }
    return evaluation

question = "What is the capital of France?"
answer = "It is Berlin"
result = evaulate_answer(question,answer)
print("Evaluation result: ", result)

            
    