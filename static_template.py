from langchain_core.prompts import PromptTemplate


static_prompt = PromptTemplate(
    input_variables = [],
    template = "write a short fun fact about knwoledge in AI"
)

prompt_text = static_prompt.format()
print(prompt_text)