# api_server.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Initialize FastAPI app
app = FastAPI(title="Style Pat Fashion Chatbot API")

# Allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

# Startup event for initializing chatbot, vector store, etc.
@app.on_event("startup")
def startup_event():
    global chatbot

    # Load OpenAI key
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("❌ OPENAI_API_KEY is not set!")

    openai_native_client = openai.OpenAI(api_key=openai_key)

    # Initialize models and components
    chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    fashion_db = FashionDatabase()
    rec_engine = FashionRecommendationEngine(fashion_db)

    # Create documents from products
    products = fashion_db.get_all_products()
    docs = []
    for product in products:
        content = f"""
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
            page_content=content.strip(),
            metadata={
                'product_id': product[0],
                'name': product[1],
                'category': product[2],
                'brand': product[4],
                'price': product[5],
                'color': product[6]
            }
        )
        docs.append(doc)

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunked_docs, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # Initialize chatbot
    chatbot = EnhancedFashionChatbot(chatgpt, retriever, fashion_db, rec_engine, openai_native_client)
    print("✅ Chatbot initialized successfully.")

# Health check route
@app.get("/")
async def root():
    return {"status": "ok"}

# Main chat endpoint
@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    try:
        image_content = await image.read() if image else None

        if image_content:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_content)
                tmp_file.flush()
                analysis_result = chatbot.handle_image_upload(user_id, tmp_file.name, message)

            response = chatbot.chat_with_image_context(
                user_id, message, image_analysis=analysis_result["analysis"]
            )
        else:
            response = chatbot.chat_with_image_context(user_id, message)

        return response

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error in /chat endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Local run (useful for testing locally)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
