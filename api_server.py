from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler

import os
import uvicorn
import tempfile
import traceback

from fashion_image_chat import EnhancedFashionChatbot, FashionDatabase, FashionRecommendationEngine
from openai import OpenAI

app = FastAPI(title="Style Pat Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    print(f"❌ Exception: {str(exc)}")
    return PlainTextResponse(f"Internal Server Error: {str(exc)}", status_code=500)

@app.on_event("startup")
def startup_event():
    global chatbot
    print("🚀 Starting chatbot...")

    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")

        openai_native_client = OpenAI(api_key=openai_key)

        # If you use retrieval/LLM, connect here, but not required for core image logic
        fashion_db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(fashion_db)
        chatbot = EnhancedFashionChatbot(
            chat_model=None,  # Optionally set your LLM
            retriever=None,   # Optionally set retrieval engine
            db=fashion_db,
            rec_engine=rec_engine,
            openai_client=openai_native_client
        )
        print("✅ Chatbot initialized successfully.")

    except Exception as ex:
        traceback.print_exc()
        print(f"❌ Startup failed: {ex}")
        raise ex

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    try:
        print(f"📩 Request from user: {user_id}")
        print(f"📝 Message: {message}")
        print(f"📷 Image attached: {image is not None}")

        image_content = None
        if image:
            try:
                image_content = await image.read()
                print(f"📥 Image read OK, {len(image_content)} bytes")
            except Exception as e:
                print(f"❌ Failed to read image: {e}")

        if image_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_content)
                tmp_file.flush()
                tmp_path = tmp_file.name
            analysis_result = chatbot.handle_image_upload(user_id, tmp_path, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis_result)
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                print(f"⚠️ Could not delete temp file {tmp_path}: {cleanup_err}")
        else:
            response = chatbot.chat_with_image_context(user_id, message)

        print("✅ Sending response")
        return response

    except Exception as ex:
        traceback.print_exc()
        print(f"❌ Error in /chat: {ex}")
        return JSONResponse({"error": str(ex)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
