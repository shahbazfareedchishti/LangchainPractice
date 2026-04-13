from langchain_core.prompts import PromptTemplate

dynamic_prompt = PromptTemplate(
    template = "write a short paragraph about {topic} in a {style} style.",
    input_variables = ["topic", "style"]
)

prompt_text = dynamic_prompt.format(
    topic = "AI", 
    style = "funny"
)

print(prompt_text)