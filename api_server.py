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
    allow_origins=["*"],  # Adjust for production best practice
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
def startup():
    global chatbot
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    openai_client = OpenAI(api_key=openai_key)
    fashion_db = FashionDatabase()
    # If you have a recommendation engine or retriever, plug them in; else keep as None
    chatbot = EnhancedFashionChatbot(
        chat_model=None,
        retriever=None,
        db=fashion_db,
        rec_engine=None,
        openai_client=openai_client
    )

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None)
):
    try:
        if image is not None:
            # Save image to a temp file for OpenAI Vision, then delete it
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[-1] or ".jpg") as tmp:
                shutil.copyfileobj(image.file, tmp)
                tmp_path = tmp.name

            analysis = chatbot.handle_image_upload(user_id, tmp_path, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis)
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: failed to delete temp file {tmp_path}: {e}")
        else:
            response = chatbot.chat_with_image_context(user_id, message)
        return response
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
