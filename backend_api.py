from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import rag_chatbot

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Academic RAG Chatbot API Running"}


@app.post("/chat")
def chat(query: Query):

    talk = rag_chatbot.small_talk(query.question)

    if talk:
        return {"answer": talk, "sources": []}

    results = rag_chatbot.retrieve(query.question)

    if len(results) == 0:
        return {
            "answer": "Sorry bro, I couldn't find an answer.",
            "sources": []
        }

    context = "\n".join(results)

    answer = rag_chatbot.generate_answer(query.question, context)

    return {
        "answer": answer,
        "sources": results[:2]
    }