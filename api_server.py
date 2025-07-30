# api_server.py

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler

import os
import uvicorn
import tempfile
import traceback

from Fashion_image_chat import (
    EnhancedFashionChatbot,
    FashionDatabase,
    FashionRecommendationEngine
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import openai


# === App Initialization ===
app = FastAPI(title="Style Pat Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None  # Global chatbot instance


# === Global Exception Handlers ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    print(f"❌ Global Exception Caught: {str(exc)}")
    return PlainTextResponse(f"Internal Server Error: {str(exc)}", status_code=500)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"❌ Validation Error: {exc}")
    return await http_exception_handler(request, exc)

@app.exception_handler(HTTPException)
async def http_exception_handler_custom(request: Request, exc: HTTPException):
    print(f"❌ HTTP Exception: {exc}")
    return await http_exception_handler(request, exc)


# === Startup Event ===
@app.on_event("startup")
def startup_event():
    global chatbot
    print("🚀 Starting chatbot...")

    try:
        # Load and check API Key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("❌ OPENAI_API_KEY environment variable is not set.")

        os.environ["OPENAI_API_KEY"] = openai_key
        openai_native_client = openai.OpenAI(api_key=openai_key)

        # Initialize chatbot components
        chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        fashion_db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(fashion_db)

        # Vector store setup
        products = fashion_db.get_all_products()
        documents = []
        for product in products:
            doc_text = f"""
            Name: {product[1]}
            Category: {product[2]} - {product[3]}
            Brand: {product[4]}
            Price: ${product[5]}
            Color: {product[6]}
            Size: {product[7]}
            Description: {product[8]}
            Style: {product[9]}
            Season: {product[10]}
            Gender: {product[11]}
            Occasion: {product[12]}
            Material: {product[13]}
            """
            doc = Document(
                page_content=doc_text.strip(),
                metadata={
                    'product_id': product[0],
                    'name': product[1],
                    'category': product[2],
                    'brand': product[4],
                    'price': product[5],
                    'color': product[6]
                }
            )
            documents.append(doc)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embedding=embed_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        # Instantiate enhanced chatbot
        chatbot = EnhancedFashionChatbot(chatgpt, retriever, fashion_db, rec_engine, openai_native_client)
        print("✅ Chatbot initialized successfully")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Startup failed: {e}")
        raise e


# === Health Check ===
@app.get("/")
async def root():
    return {"status": "ok"}


# === Chat Endpoint ===
@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    try:
        print(f"📩 Incoming request from user: {user_id}")
        print(f"📝 Message: {message}")
        print(f"📷 Image received: {image is not None}")

        image_content = None
        if image:
            try:
                image_content = await image.read()
                print(f"📥 Image read successful — size: {len(image_content)} bytes")
            except Exception as read_err:
                print("❌ Failed to read image:", read_err)

        # If image was uploaded, analyze and use it
        if image_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_content)
                tmp_file.flush()
                tmp_path = tmp_file.name
                print(f"🖼️ Image saved to: {tmp_path}")

            analysis_result = chatbot.handle_image_upload(user_id, tmp_path, message)

            if not analysis_result.get("success", False):
                print(f"⚠️ Image analysis failed: {analysis_result.get('error')}")
                return JSONResponse({"error": "Image analysis failed."}, status_code=500)

            response = chatbot.chat_with_image_context(
                user_id, message, image_analysis=analysis_result.get("analysis", {})
            )
        else:
            print("💬 No image attached — text-only message.")
            response = chatbot.chat_with_image_context(user_id, message)

        print("✅ Chat response ready.")
        return response

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error in /chat endpoint: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


# === Main Entry Point for Local Dev ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
