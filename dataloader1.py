from langchain_community.document_loaders import TextLoader

loader = TextLoader("islamabad.txt", encoding = "utf-8")

docs = loader.load()

print(docs)
print(type(docs))
print(docs[0])
print(docs[0].metadata)