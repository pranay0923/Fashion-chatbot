from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your chatbot classes and initialization from fashion_chatbot.py
# Assuming you saved your chatbot code in fashion_chatbot.py with a FashionChatbot class

from fashion_chatbot import FashionChatbot, FashionDatabase, FashionRecommendationEngine, ChatOpenAI, OpenAIEmbeddings, Chroma

import os

app = FastAPI(title="Fashion Chatbot API")

# Allow CORS for local dev or your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize backend components once on startup (do expensive loading here)
@app.on_event("startup")
def startup_event():
    global chatbot

    # Load OpenAI API key from env
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    chatgpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    fashion_db = FashionDatabase()
    # Optionally load fashion data here if needed, or assume data already exists

    # Build vector DB
    fashion_docs = []
    import sqlite3
    conn = sqlite3.connect(fashion_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    conn.close()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document

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
        collection_name='fashion_db',
        embedding=openai_embed_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./fashion_vector_db"
    )

    fashion_retriever = fashion_db_vector.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.3}
    )

    rec_engine = FashionRecommendationEngine(fashion_db, fashion_retriever)
    chatbot = FashionChatbot(chatgpt, fashion_retriever, fashion_db, rec_engine)

# Define request schema for text messages only (optional)
class ChatRequest(BaseModel):
    user_id: str
    message: str = ""  # can be empty if user sends only image

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None)
):
    """
    Accept a chat message and optional image from user,
    return assistant's answer.
    """

    # Optionally: process or store image - For now just an acknowledgment
    # You can read image bytes if needed:
    if image:
        image_content = await image.read()
        # Here you can add vision processing or other logic, or
        # add mention of image received to the prompt/context if you like.

    # Call your chatbot logic
    response = chatbot.chat(user_id, message)

    # Optionally add image acknowledgement in response if image sent
    if image:
        response["answer"] += "\n\n[Note: Image received and processed.]"

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
