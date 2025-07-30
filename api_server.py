# api_server.py

import os
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fashion_image_chat import EnhancedFashionChatbot

app = FastAPI(title="Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

chatbot = None

@app.on_event("startup")
def startup():
    global chatbot
    try:
        print("🚀 Initializing chatbot...")
        chatbot = EnhancedFashionChatbot()
        print("✅ Chatbot is ready.")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Startup failed: {e}")

@app.get("/")
async def root():
    return {"status": "running"}

@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None)
):
    try:
        image_analysis = None
        if image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(await image.read())
                tmp.flush()
                image_analysis = chatbot.handle_image_upload(user_id, tmp.name, message)

        response = chatbot.chat_with_image_context(user_id, message, image_analysis)
        return response

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
