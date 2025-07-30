from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import base64

from fashion_image_chat import (
    setup_openai_api,
    FashionDatabase,
    FashionRecommendationEngine,
    create_fashion_vector_store,
    EnhancedFashionChatbot
)

# ==== Step 1: FastAPI Setup ====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Step 2: Global Chatbot Initialization ====
chatgpt, openai_embed_model, openai_native_client = setup_openai_api()

fashion_db = FashionDatabase()
rec_engine = FashionRecommendationEngine(fashion_db)
vectorstore = create_fashion_vector_store(fashion_db, openai_embed_model)
fashion_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

chatbot = EnhancedFashionChatbot(
    chatgpt, fashion_retriever, fashion_db, rec_engine, openai_native_client
)

# ==== Step 3: Routes ====

@app.get("/")
def read_root():
    return {"message": "Fashion Chatbot API is running. Use the /chat endpoint to interact."}


@app.post("/chat")
async def chat_with_fashion_bot(user_id: str = Form(...), message: str = Form(...)):
    try:
        response = chatbot.chat_with_image_context(user_id, message)
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze-image")
async def analyze_image(user_id: str = Form(...), query: str = Form("Analyze this fashion image"), file: UploadFile = File(...)):
    try:
        # Save uploaded image to uploads/ directory
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_location = os.path.join(upload_dir, file.filename)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image and analyze it
        result = chatbot.handle_image_upload(user_id, file_location, query=query)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

