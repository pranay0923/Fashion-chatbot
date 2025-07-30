import os
import tempfile
import shutil
import uvicorn
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openai import OpenAI
from fashion_image_chat import EnhancedFashionChatbot, FashionDatabase

app = FastAPI(title="Fashion AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
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
async def chat(user_id: str = Form(...), message: str = Form(""), image: UploadFile = File(None)):
    try:
        if image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[-1]) as tmp:
                shutil.copyfileobj(image.file, tmp)
                tmp_path = tmp.name
            analysis = chatbot.handle_image_upload(user_id, tmp_path, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis)
            os.unlink(tmp_path)
        else:
            response = chatbot.chat_with_image_context(user_id, message)
        return response
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
