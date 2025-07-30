# api_server.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import base64
import tempfile

from fashion_image_chat import (
    FashionDatabase,
    FashionRecommendationEngine,
    create_fashion_vector_store,
    EnhancedFashionChatbot,
    setup_openai_api
)

# --- Step 1: FastAPI App Setup ---
app = FastAPI()

# Allow frontend from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Step 2: Initialize Components ---
chatgpt, embed_model, openai_client = setup_openai_api()
fashion_db = FashionDatabase()
rec_engine = FashionRecommendationEngine(fashion_db)
vectorstore = create_fashion_vector_store(fashion_db, embed_model)
fashion_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
fashion_bot = EnhancedFashionChatbot(chatgpt, fashion_retriever, fashion_db, rec_engine, openai_client)

# --- Step 3: Request & Response Schemas ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_base64: str = None

class ChatResponse(BaseModel):
    answer: str
    recommendations: list
    context_products: list
    user_preferences: dict
    image_analysis: dict | None

# --- Step 4: Health Check ---
@app.get("/")
def root():
    return {"message": "Fashion Chatbot API is running. Use the /chat endpoint to interact."}

# --- Step 5: Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_fashion_bot(request: ChatRequest):
    image_analysis = None

    # Handle base64 image if present
    if request.image_base64:
        # Save temporary image
        file_bytes = base64.b64decode(request.image_base64.split(",")[-1])
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        result = fashion_bot.handle_image_upload(
            user_id=request.user_id,
            image_file_path=temp_path,
            query=request.message
        )
        if result.get("success"):
            image_analysis = result.get("analysis")

    # Get chatbot response
    response = fashion_bot.chat_with_image_context(
        user_id=request.user_id,
        message=request.message,
        image_analysis=image_analysis
    )

    return ChatResponse(**response)
