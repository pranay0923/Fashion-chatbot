import os
import tempfile
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

from openai import OpenAI
from fashion_image_chat import EnhancedFashionChatbot, FashionDatabase

app = FastAPI(title="Fashion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production restrict this appropriately
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)

    # Initialize DB and chatbot
    db = FashionDatabase()
    rec_engine = None  # Implement and pass your recommendation engine if available
    chatbot_instance = EnhancedFashionChatbot(chat_model=None, retriever=None, db=db, rec_engine=rec_engine, openai_client=openai_client)

    chatbot = chatbot_instance
    print("Chatbot initialized successfully")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    try:
        if image:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(image.file, tmp)
                tmp_path = tmp.name

            analysis = chatbot.handle_image_upload(user_id, tmp_path, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis)

            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: failed to remove tmp file {tmp_path}: {e}")

            return response
        else:
            response = chatbot.chat_with_image_context(user_id, message)
            return response

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port)
