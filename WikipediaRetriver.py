from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results = 2,
    lang = "en"
)

query = "Artificial Intelligence"

docs = retriever.invoke(query)

print(len(docs))

for i, doc in enumerate(docs):
    print("content:\n", doc.page_content)