from langchain_text_splitters import PythonCodeTextSplitter, RecursiveCharacterTextSplitter, Language

text = """
# ... (rest of your imports and model definitions stay the same)

# 1. Update your prompts to use a consistent variable name (optional but cleaner)
prompt1 = PromptTemplate(template="Write a short note from: {input_text}", input_variables=["input_text"])
prompt2 = PromptTemplate(template="Generate 5 short questions from: {input_text}", input_variables=["input_text"])

# 2. Re-structure the RunnableParallel to pull from ONE input key
runnable_chain = RunnableParallel(
    {
        # We tell LangChain: take the 'input_text' from the dictionary and pass it to both
        'notes': (lambda x: {"input_text": x["input_text"]}) | prompt1 | model1 | parser,
        'qa': (lambda x: {"input_text": x["input_text"]}) | prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

final_chain = runnable_chain | merge_chain

# 3. Invoke with a single key
result = final_chain.invoke({"input_text": text_to_process})
print(result)
"""

splitter = PythonCodeTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
)

chunks = splitter.split_text(text)

print(chunks)