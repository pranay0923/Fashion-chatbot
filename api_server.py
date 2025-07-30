import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from openai import OpenAI

from fashion_image import (
    EnhancedFashionChatbot,
    FashionDatabase,
    FashionRecommendationEngine,
)

app = FastAPI(title="Fashion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
def startup():
    global chatbot

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    openai_client = OpenAI(api_key=api_key)

    # Init LangChain chat model and embed model
    chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
    embed_model = OpenAIEmbeddings(model="text-embedding-3")

    # Load or init DB
    db = FashionDatabase()

    # Prepare vector DB for products (load or build embedding)
    products = db.get_all_products()
    from langchain.vectorstores import FAISS

    docs = []
    for p in products:
        txt = f"Product: {p[1]}, Category: {p}, Brand: {p}, Color: {p}, Price: {p}, Description: {p}"
        docs.append(Document(page_content=txt, metadata={"product_id": p}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embed_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    rec_engine = FashionRecommendationEngine(db, retriever)

    chatbot = EnhancedFashionChatbot(chat_model, retriever, db, rec_engine, openai_client)

@app.get("/")
def get_root():
    return {"status": "ok"}

@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(default=""),
    image: UploadFile = File(default=None),
):
    try:
        if image is not None:
            # Save image temporary
            suffix = os.path.splitext(image.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(image.file, tmp)
                tmp_path = tmp.name

            analysis = chatbot.handle_uploaded_image(user_id, open(tmp_path, "rb"), message)
            response = chatbot.chat_with_context(user_id, message, image_analysis=analysis)
            os.unlink(tmp_path)
        else:
            response = chatbot.chat_with_context(user_id, message)

        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
