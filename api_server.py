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

# === FastAPI App Initialization ===
app = FastAPI(title="Style Pat Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ You can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global Chatbot Variable ===
chatbot = None

# === Startup Logic ===
@app.on_event("startup")
def startup_event():
    global chatbot
    print("🚀 Starting chatbot...")

    try:
        # --- Load API Key ---
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("❌ OPENAI_API_KEY not set!")

        os.environ["OPENAI_API_KEY"] = openai_key
        openai_native_client = openai.OpenAI(api_key=openai_key)

        # --- Initialize Models & DB ---
        chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        fashion_db = FashionDatabase()
        rec_engine = FashionRecommendationEngine(fashion_db)

        # --- Build Vector Store ---
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

        # --- Initialize Chatbot ---
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

# === Main Chat Endpoint ===
@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    try:
        print(f"📩 User ID: {user_id}")
        print(f"📝 Message: {message}")
        image_content = await image.read() if image else None
        has_image = bool(image_content)
        print(f"📷 Image attached: {has_image}")

        # If image uploaded
        if has_image:
            # Save image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_content)
                tmp_file.flush()
                tmp_path = tmp_file.name

            print(f"🖼️ Image saved to temp path: {tmp_path}")

            # Analyze the image
            analysis_result = chatbot.handle_image_upload(user_id, tmp_path, message)

            if not analysis_result.get("success", False):
                error_message = analysis_result.get("error", "Unknown image error")
                print(f"⚠️ Image analysis failed: {error_message}")
                return JSONResponse({"error": f"Image processing failed: {error_message}"}, status_code=500)

            # Generate response with image context
            response = chatbot.chat_with_image_context(
                user_id, message, image_analysis=analysis_result.get("analysis", {})
            )
            print("✅ Response generated with image context")
        else:
            # Generate response without image
            response = chatbot.chat_with_image_context(user_id, message)
            print("✅ Response generated without image")

        return response

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error in /chat endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === Local Dev Mode ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
