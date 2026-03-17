from sentence_transformers import SentenceTransformer
from pdf_loader import load_all_pdfs
from text_chunker import chunk_text
import faiss
import numpy as np
import pickle


# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


def build_vector_store(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


def save_vector_store(index):
    faiss.write_index(index, "vector_store.index")


if __name__ == "__main__":

    # load pdf text
    text = load_all_pdfs("../data")

    # chunk text
    chunks = chunk_text(text)

    # create embeddings
    embeddings = create_embeddings(chunks)

    # build vector database
    index = build_vector_store(embeddings)

    # save vector database
    save_vector_store(index)

    # save chunks so we can retrieve answers later
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Vector database created successfully")