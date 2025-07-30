import os
import tempfile
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

from fashion_image_chat import EnhancedFashionbot, FashionDatabase, FashionRecommendationEngine
from openai import OpenAI

app = FastAPI(title="Fashion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None  # Will be initialized in startup event


@app.on_event("startup")
def startup_event():
    global chatbot

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=openai_api_key)

    # Initialize your database and chatbot here
    db = FashionDatabase()
    # If you have a vector retriever ready, assign it, else None
    retriever = None

    rec_engine = FashionRecommendationEngine(db, retriever)

    chatbot = EnhancedFashionbot(
        chat_model=None,  # Replace with your chat model if available (e.g., ChatOpenAI instance)
        retriever=retriever,
        db=db,
        rec_engine=rec_engine,
        openai_client=client,
    )

    print("Chatbot started successfully")


@app.get("/")
def get_root():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(default=""),
    image: UploadFile = File(default=None),
):
    try:
        if image is not None:
            # Save uploaded image to a temporary file
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(image.file, tmp_file)
                temp_path = tmp_file.name

            # Call chatbot image processing
            analysis_result = chatbot.handle_image_upload(user_id, temp_path, message)
            # Get chat reply with image context
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis_result)
            # Delete temp file
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: unable to delete temp file {temp_path}: {e}")

        else:
            # Text-only chat
            response = chatbot.chat_with_image_context(user_id, message)

        return response

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port)
