import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vector_store.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve(query, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]

    return results