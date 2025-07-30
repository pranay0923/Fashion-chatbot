from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn
import tempfile

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

app = FastAPI(title="Style Pat Fashion Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot

    # 1. Set keys
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    openai_native_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 2. Initialize models and DB
    chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
    openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    fashion_db = FashionDatabase()
    rec_engine = FashionRecommendationEngine(fashion_db)

    # 3. Build Vector Store (FAISS)
    products = fashion_db.get_all_products()
    docs = []
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
        docs.append(doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunked_docs, embedding=openai_embed_model)
    fashion_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # 4. Initialize ENHANCED chatbot
    chatbot = EnhancedFashionChatbot(
        chatgpt, fashion_retriever, fashion_db, rec_engine, openai_native_client
    )

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
        image_content = await image.read() if image else None
        if image_content:
            # Save uploaded image to temp path for analysis
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_content)
                tmp.flush()
                img_analysis = chatbot.handle_image_upload(user_id, tmp.name, message)
            response = chatbot.chat_with_image_context(user_id, message, image_analysis=img_analysis['analysis'])
        else:
            response = chatbot.chat_with_image_context(user_id, message)
        return response
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
