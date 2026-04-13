from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate 

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template( "You are a helpful assistant that provides information about {subject}."),
    HumanMessagePromptTemplate.from_template("Can you tell me something about {subject}")
])

prompt = chat_prompt.format_messages(subject =  "AI")

print(prompt)