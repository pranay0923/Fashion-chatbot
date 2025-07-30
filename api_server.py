from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler

import os
import uvicorn
import tempfile
import traceback

from fashion_image_chat import (
    EnhancedFashionChatbot,
    FashionDatabase,
    FashionRecommendationEngine
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

chatbot = None

# === Global Exception Handlers ===
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

# === Startup Event ===
@app.on_event("startup")
def startup_event():
    global chatbot
    print("🚀 Starting chatbot...")

    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("❌ OPENAI_API_KEY environment variable is not set.")
        os.environ["OPENAI_API_KEY"] = openai_key
        openai_native_client = openai.OpenAI(api_key=openai_key)

        chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        fashion_db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(fashion_db)
        # Simple/mocked vector store & retriever
        docs = [Document(page_content="Denim Jeans - A classic choice.", metadata={"product_id":1})]
        vectorstore = FAISS.from_documents(docs, embedding=embed_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        chatbot = EnhancedFashionChatbot(chatgpt, retriever, fashion_db, rec_engine, openai_native_client)
        print("✅ Chatbot initialized successfully")
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
        response = None
        image_temp_path = None
        if image:
            try:
                image_content = await image.read()
                if image_content:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(image_content)
                        tmp_file.flush()
                        image_temp_path = tmp_file.name
            except Exception as read_err:
                return JSONResponse({"answer": f"Could not read image: {str(read_err)}"}, status_code=400)

        if image_temp_path:
            analysis_result = chatbot.handle_image_upload(user_id, image_temp_path, message)
            try:
                os.remove(image_temp_path)
            except Exception:
                pass
            if not analysis_result.get("success", False):
                return JSONResponse({"answer": f"Image analysis failed: {analysis_result.get('error', 'unknown error')}"}, status_code=500)
            result = chatbot.chat_with_image_context(user_id, message, image_analysis=analysis_result.get("analysis", {}))
        else:
            result = chatbot.chat_with_image_context(user_id, message)

        return JSONResponse({
            "answer": result.get("answer", "🤔 I don't know how to respond to that."),
            "image_analysis": result.get("image_analysis", {}),
            "user_id": user_id
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"answer": f"Server error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
