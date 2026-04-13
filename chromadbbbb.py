from langchain_chroma import Chroma  # Capital C
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

texts = [
    "Vector databases perform similarity searches using distance metrics like cosine similarity",
    "Quantization reduces model size by decreasing the precision of weight tensors",
    "Retrieval-Augmented Generation mitigates hallucinations by grounding models in external data",
    "Attention mechanisms allow models to weigh the importance of different tokens in a sequence",
    "Fine-tuning adjusts pre-trained weights to specialize a model for specific vertical domains",
    "Context windows are finite and constrained by the quadratic scaling of self-attention",
    "Tokenization is the often-overlooked bottleneck in how models perceive linguistic nuance",
    "Prompt engineering is a temporary heuristic for poorly aligned model behavior"
]

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Use Chroma (Capital C) and the correct parameter name 'texts'
vectorstore = Chroma.from_texts(
    texts = texts, 
    embedding = embedding,
    collection_name = "langchain_chroma_demo"
)

query = "tell me more about LLMs"
results = vectorstore.similarity_search(query, k=2)

print(results)