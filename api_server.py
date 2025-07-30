import os
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler

from Fashion_image_chat import (
    EnhancedFashionChatbot,
    FashionDatabase,
    FashionRecommendationEngine,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import openai

app = FastAPI(title="Style Pat Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None  # Global chatbot instance


# === Exception Handlers ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return PlainTextResponse(f"Internal Server Error: {str(exc)}", status_code=500)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return await http_exception_handler(request, exc)

@app.exception_handler(HTTPException)
async def http_exception_handler_custom(request: Request, exc: HTTPException):
    return await http_exception_handler(request, exc)


@app.on_event("startup")
def startup_event():
    global chatbot
    print("🚀 Starting chatbot...")

    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("❌ OPENAI_API_KEY not set")

        os.environ["OPENAI_API_KEY"] = openai_key
        openai_client = openai.OpenAI(api_key=openai_key)

        # LangChain Setup
        chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

        db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(db)

        # Create product documents for vector store
        products = db.get_all_products()
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
            documents.append(Document(
                page_content=doc_text.strip(),
                metadata={"product_id": product[0], "brand": product[4], "name": product[1]}
            ))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embedding=embed_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        chatbot = EnhancedFashionChatbot(chat_model, retriever, db, rec_engine, openai_client)
        print("✅ Chatbot initialized.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Startup failed: {e}")
        raise e


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
        print(f"📩 User: {user_id}")
        print(f"📝 Message: {message}")
        print(f"📷 Image present: {image is not None}")

        image_content = await image.read() if image else None

        if image_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_content)
                tmp.flush()
                temp_path = tmp.name

            analysis = chatbot.handle_image_upload(user_id, temp_path, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis)

            os.remove(temp_path)
        else:
            response = chatbot.chat_with_image_context(user_id, message)

        return response

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
