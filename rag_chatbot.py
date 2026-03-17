import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# GROQ CLIENT
# ------------------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing bro!")

client = Groq(api_key=api_key)

 

# ------------------------------
# EMBEDDING MODEL
# ------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# LOAD VECTOR DATABASE
# ------------------------------

index = faiss.read_index("vector_store.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# ------------------------------
# RETRIEVE RELEVANT CHUNKS
# ------------------------------

def retrieve(query, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:

        if i < len(chunks):

            text = chunks[i]

            text = text.replace("\n", " ")

            results.append(text.strip())

    return results


# ------------------------------
# GENERATE ANSWER USING GROQ
# ------------------------------

def generate_answer(query, context):

    prompt = f"""
You are an academic assistant.

Context:
{context}

Question:
{query}

Provide a clear and short explanation.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
# ------------------------------
# SMALL TALK HANDLING
# ------------------------------

def small_talk(text):

    text = text.lower().strip()

    if text in ["hi","hello","hey"]:
        return "Hello bro! How can I help you?"

    if text in ["thanks","thank you"]:
        return "You're welcome bro!"

    if text in ["bye"]:
        return "Goodbye!"

    return None