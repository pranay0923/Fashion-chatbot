import os
import uvicorn
import tempfile
import traceback
import requests
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, HTTPException

from fashion_image_chat import EnhancedFashionChatbot, FashionDatabase, FashionRecommendationEngine
from openai import OpenAI

app = FastAPI(title="Style Pat Fashion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return PlainTextResponse(f"Internal Server Error: {exc}", status_code=500)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc}")
    return await http_exception_handler(request, exc)

@app.exception_handler(HTTPException)
async def http_exception_handler_custom(request: Request, exc: HTTPException):
    print(f"HTTP exception: {exc}")
    return await http_exception_handler(request, exc)

@app.on_event("startup")
def startup_event():
    global chatbot
    print("Starting chatbot engine...")

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")

        openai_client = OpenAI(api_key=openai_api_key)

        fashion_db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(fashion_db)

        # You can set chat_model and retriever to None or real instances if ready
        chatbot = EnhancedFashionChatbot(
            chat_model=None,
            retriever=None,
            db=fashion_db,
            rec_engine=rec_engine,
            openai_client=openai_client,
        )
        print("Chatbot initialized successfully")

    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        raise

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
        print(f"Request from user: {user_id}, message: {message}")
        if image:
            image_bytes = await image.read()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(image_bytes)
            temp_file.flush()
            temp_file.close()

            print(f"Saved uploaded image to {temp_file.name}")

            analysis = chatbot.handle_image_upload(user_id, temp_file.name, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis)

            try:
                os.remove(temp_file.name)
            except Exception as e:
                print(f"Warning: failed to delete temp image file {temp_file.name}: {e}")
        else:
            response = chatbot.chat_with_image_context(user_id, message)

        return response

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
