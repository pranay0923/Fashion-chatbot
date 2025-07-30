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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import openai


app = FastAPI(title="Style Pat Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


chatbot = None  # Global chatbot instance


# --- Exception Handlers ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    print(f"❌ Global Exception: {str(exc)}")
    return PlainTextResponse(f"Internal Server Error: {str(exc)}", status_code=500)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"❌ Validation Error: {exc}")
    return await http_exception_handler(request, exc)

@app.exception_handler(HTTPException)
async def http_exception_handler_custom(request: Request, exc: HTTPException):
    print(f"❌ HTTP Exception: {exc}")
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
        openai_native_client = openai.OpenAI(api_key=openai_key)

        # Model init
        chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        fashion_db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(fashion_db)

        # Vector store preparation
        products = fashion_db.get_all_products()
        documents = []
        for product in products:
            doc_text = f"""
            Name: {product}
            Category: {product} - {product}
            Brand: {product}
            Price: ${product}
            Color: {product}
            Size: {product}
            Description: {product}
            Style: {product}
            Season: {product}
            Gender: {product}
            Occasion: {product}
            Material: {product}
            """
            doc = Document(
                page_content=doc_text.strip(),
                metadata={
                    'product_id': product,
                    'name': product,
                    'category': product,
                    'brand': product,
                    'price': product,
                    'color': product
                }
            )
            documents.append(doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embedding=embed_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        chatbot = EnhancedFashionChatbot(chatgpt, retriever, fashion_db, rec_engine, openai_native_client)
        print("✅ Chatbot initialized successfully.")
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
        print(f"📩 Request from user: {user_id}")
        print(f"📝 Message: {message}")
        print(f"📷 Image attached: {image is not None}")

        image_content = None
        if image:
            try:
                image_content = await image.read()
                print(f"📥 Image read success: {len(image_content)} bytes")
            except Exception as err:
                print(f"❌ Failed to read image: {err}")

        if image_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(image_content)
                temp_file.flush()
                temp_path = temp_file.name
                print(f"🖼️ Image temporarily saved: {temp_path}")

            analysis_result = chatbot.handle_image_upload(user_id, temp_path, message)

            # Pass analysis result directly (per updated chatbot design)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis_result)

            # Clean up temp file
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"⚠️ Could not delete temp file {temp_path}: {cleanup_err}")
        else:
            print("💬 Text-only input, no image.")
            response = chatbot.chat_with_image_context(user_id, message)

        print("✅ Response ready to send back.")
        return response

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error during /chat: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
