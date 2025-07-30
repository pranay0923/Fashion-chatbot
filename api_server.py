# api_server.py

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fashion_image_chat import FashionImageChatHandler
from io import BytesIO

# Initialize app and handler
app = FastAPI()
fashion_handler = FashionImageChatHandler()

# CORS setup (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Fashion Chatbot API is running. Use the /chat endpoint to interact."}


@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = None
):
    try:
        if image:
            # Wrap UploadFile bytes into BytesIO
            image_file = BytesIO(await image.read())
            image_file.name = image.filename
            image_file.type = image.content_type

            result = fashion_handler.process_image_and_chat(user_id, image_file, message)
            return JSONResponse(content=result)
        else:
            return JSONResponse(
                content={"answer": "🗨️ You didn't upload an image. Please upload a fashion image."},
                status_code=200
            )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "message": "Something went wrong processing your request."},
            status_code=500
        )
