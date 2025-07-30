from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sqlite3
from fashion_chatbot import (
    FashionChatbot,
    FashionDatabase,
    FashionRecommendationEngine,
    ChatOpenAI,
    OpenAIEmbeddings,
    Chroma,
    RecursiveCharacterTextSplitter,
    Document,
)

import uvicorn

app = FastAPI(title="Fashion Chatbot API")

# CORS setup - allow your frontend domain here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend origin in production, e.g. https://yourdomain.com
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot variable
chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    chatgpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    fashion_db = FashionDatabase()
    # Load fashion data if necessary (optional)
    # load_fashion_data(fashion_db)  # Uncomment if you have a loader function

    # Build vector DB from DB products
    fashion_docs = []
    conn = sqlite3.connect(fashion_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    conn.close()

    for product in products:
        content = f"""
        Product: {product[1]}
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
            page_content=content,
            metadata={
                "product_id": product[0],
                "name": product[1],
                "category": product[2],
                "brand": product[4],
                "price": product[5],
            }
        )
        fashion_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = splitter.split_documents(fashion_docs)

    fashion_db_vector = Chroma.from_documents(
        documents=chunked_docs,
        collection_name="fashion_db",
        embedding=openai_embed_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./fashion_vector_db",
    )

    fashion_retriever = fashion_db_vector.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.3},
    )

    rec_engine = FashionRecommendationEngine(fashion_db, fashion_retriever)
    chatbot = FashionChatbot(chatgpt, fashion_retriever, fashion_db, rec_engine)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    """
    Accept both text message and optional image upload in multipart form.
    Process the chat query and optionally the image.
    """

    try:
        # If an image is uploaded, read bytes (expand your processing if needed)
        if image is not None:
            image_content = await image.read()
            # For now, just log or acknowledge image upload; extend with vision tasks if you want
            # e.g., fashion attributes extraction here
        else:
            image_content = None

        # Invoke chatbot logic. You can extend FashionChatbot.chat to optionally accept image bytes if desired.
        response = chatbot.chat(user_id, message)  # Current chatbot interface doesn't use image_content

        # If image received, optionally append info
        if image_content:
            response[
                "answer"
            ] += "\n\n[Note: Image received and can be processed for fashion analysis.]"

        return response

    except Exception as e:
        # Log error details for debugging
        print(f"Error in /chat endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    # You can override host/port by env vars if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
