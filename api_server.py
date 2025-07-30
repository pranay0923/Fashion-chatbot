# api_server.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from Fashion_image_chat import get_fashion_chatbot

app = FastAPI(title="Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY env variable must be set.")
    chatbot = get_fashion_chatbot(openai_key)
    print("Chatbot initialized and ready.")


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None)
):
    try:
        image_content = await image.read() if image else None
        response = chatbot.chat(user_id, message, image_bytes=image_content)
        return JSONResponse(content=response)
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
