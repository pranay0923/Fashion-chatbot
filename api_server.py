from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, sqlite3

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    chatgpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    fashion_db = FashionDatabase()
    # ... build vector db
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
    chatbot_obj = FashionChatbot(chatgpt, fashion_retriever, fashion_db, rec_engine)
    global chatbot
    chatbot = chatbot_obj

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
        image_content = await image.read() if image is not None else None
        response = chatbot.chat(user_id, message, image_bytes=image_content)
        return response
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
