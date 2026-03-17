from backend_api import app
@app.get("/")
def home():
    return {"message": "Backend running 🚀"}
