from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
import os

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0.7
)

class Myclass(TypedDict):
    key_themes: Annotated[list[str], "must write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "must write down a brief summary of the review"]
    sentiment: Annotated[str, "must return sentiment of the review, either Positive or Negative"]
    pros: Annotated[Optional[list[str]], "write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "write down all the cons inside a list"]

structured_llm = llm.with_structured_output(Myclass)

prompt = """
    The Samsung Galaxy A57 is a masterclass in incremental refinement, prioritizing a sleek, premium feel over radical hardware shifts. Its standout achievement is the physical redesign; by shaving nearly 20g off its predecessor and slimming down to just 6.9mm, it is remarkably comfortable to hold, though the return of the prominent camera rings feels like a slight aesthetic regression. The 6.7-inch Super AMOLED display remains gorgeous with a 120Hz refresh rate, now framed by significantly thinner bezels that finally rival flagship aesthetics. Under the hood, the Exynos 1680 chip paired with One UI 8.5 provides a fluid experience, though Awesome Intelligence (Samsung's mid-range AI suite) lacks the full depth of the S-series features. While the 50MP triple-camera system delivers reliable daylight shots and the 5000mAh battery with 45W charging remains solid, the lack of a dedicated telephoto lens and the steep price hike—now starting around 56,999 INR—makes it a tough sell against more aggressive competitors, unless you value Samsung’s superior six-year software commitment and IP68 durability above all else.
"""
response = structured_llm.invoke(prompt)
print(response)