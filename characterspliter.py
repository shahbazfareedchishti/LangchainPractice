from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("text.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separator = ""
)

result = splitter.split_documents(docs)

print(result[0].page_content)
print(result[0].metadata)
