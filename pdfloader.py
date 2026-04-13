from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("resume_shahbazfareed.pdf")

docs = loader.load()

print(docs[0].metadata)