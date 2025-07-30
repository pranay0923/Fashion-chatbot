import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import traceback

from Fashion_image_chat import (
    EnhancedFashionChatbot,
    FashionDatabase,
    FashionRecommendationEngine,
    setup_openai_api,  # The function you wrote for model setup
)

app = FastAPI(title="Fashion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot
    # Load OpenAI key from environment for security
    chatgpt, openai_embed_model, openai_native_client = setup_openai_api()
    db = FashionDatabase()
    rec_engine = FashionRecommendationEngine(db)
    # Rec engine can be enhanced to use retriever/vector search if desired
    chatbot = EnhancedFashionChatbot(
        chatgpt, None, db, rec_engine, openai_native_client
    )

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(''),
    image: UploadFile = File(None)
):
    try:
        if image:
            suffix = os.path.splitext(image.filename)[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(image.file, tmp)
                tmp_path = tmp.name
            analysis = chatbot.handle_image_upload(user_id, tmp_path, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis)
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: could not delete temp file: {e}")
        else:
            response = chatbot.chat_with_image_context(user_id, message)
        return response
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
