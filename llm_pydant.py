from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field


load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile", 
    groq_api_key = os.getenv("GROQ_API_KEY"), 
    temperature = 0.7
)

class MyClass(BaseModel):
    key_themes: list[str] = Field(description= "Write down the key theme discussed in the review in a list")
    summary: str = Field(description= "Write down a brief summary of the review")
    sentiment: Literal["Positive", "Negative"] = Field(description= "Write down the sentiment of the review, either Positive or Negative")
    pros: Optional[list[str]] = Field(default = None, description= "Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description= "Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description= "Write down the name of the product")

    
structured_llm = llm.with_structured_output(MyClass, strict = True)
prompt = """
    The Samsung Galaxy A57 is a masterclass in incremental refinement, prioritizing a sleek, premium feel over radical hardware shifts. Its standout achievement is the physical redesign; by shaving nearly 20g off its predecessor and slimming down to just 6.9mm, it is remarkably comfortable to hold, though the return of the prominent camera rings feels like a slight aesthetic regression. The 6.7-inch Super AMOLED display remains gorgeous with a 120Hz refresh rate, now framed by significantly thinner bezels that finally rival flagship aesthetics. Under the hood, the Exynos 1680 chip paired with One UI 8.5 provides a fluid experience, though Awesome Intelligence (Samsung's mid-range AI suite) lacks the full depth of the S-series features. While the 50MP triple-camera system delivers reliable daylight shots and the 5000mAh battery with 45W charging remains solid, the lack of a dedicated telephoto lens and the steep price hike—now starting around 56,999 INR—makes it a tough sell against more aggressive competitors, unless you value Samsung’s superior six-year software commitment and IP68 durability above all else.
"""
response = structured_llm.invoke(prompt)
print(response)