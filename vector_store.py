import faiss
import numpy as np

from pdf_loader import load_all_pdfs
from text_chunker import chunk_text
from embeddings import create_embeddings


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


if __name__ == "__main__":

    # load pdf text
    text = load_all_pdfs("../data")

    # split into chunks
    chunks = chunk_text(text)

    # create embeddings
    embeddings = create_embeddings(chunks)

    embeddings = np.array(embeddings)

    # build vector index
    index = build_faiss_index(embeddings)

    print("Total vectors stored:", index.ntotal)