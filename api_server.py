from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler

import os
import uvicorn
import tempfile
import traceback

from Fashion_image_chat import EnhancedFashionChatbot, FashionDatabase, FashionRecommendationEngine
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import openai

# === FastAPI App Initialization ===
app = FastAPI(title="Fashion AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Limit this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None  # Global chatbot instance

# === Exception Handling ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
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
    print("🚀 Initializing Fashion Chatbot...")

    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("❌ OPENAI_API_KEY not set in environment.")

        os.environ["OPENAI_API_KEY"] = openai_key
        openai_native_client = openai.OpenAI(api_key=openai_key)

        # Model & DB
        chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(db)

        # Vector store using product catalog
        documents = []
        for product in db.get_all_products():
            doc_text = f"""
            Name: {product[1]}
            Category: {product[2]}
            Type: {product[3]}
            Brand: {product[4]}
            Price: {product[5]}
            Color: {product[6]}
            Size: {product[7]}
            Description: {product[8]}
            Style: {product[9]}
            Season: {product[10]}
            Gender: {product[11]}
            Occasion: {product[12]}
            Material: {product[13]}
            """
            documents.append(Document(page_content=doc_text.strip(), metadata={"product_id": product[0]}))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embedding=embed_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        chatbot = EnhancedFashionChatbot(chat_model, retriever, db, rec_engine, openai_native_client)
        print("✅ Chatbot initialized successfully.")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"❌ Failed to initialize chatbot: {e}")


# === Root Endpoint ===
@app.get("/")
async def root():
    return {"message": "Fashion Chatbot API is running. Use the /chat endpoint to interact."}


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
            image_content = await image.read()
            print(f"📥 Image size: {len(image_content)} bytes")

        if image_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_content)
                tmp.flush()
                temp_path = tmp.name
            print(f"🖼️ Saved image to: {temp_path}")

            analysis = chatbot.handle_image_upload(user_id, temp_path, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis)

            os.remove(temp_path)
        else:
            print("💬 No image provided, text-only interaction.")
            response = chatbot.chat_with_image_context(user_id, message)

        print("✅ Response ready.")
        return response

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# === Run Locally ===
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
