from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
import uuid
import os
from Fashion_image_chat import ImageAnalysisService
from openai import OpenAI

# --- Setup ---
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Load API Key from env ---
openai_key = os.getenv("OPENAI_API_KEY", "your-openai-key")
openai_client = OpenAI(api_key=openai_key)
image_analyzer = ImageAnalysisService(openai_client)

# --- Request Schema ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_base64: str = None

# --- Endpoint ---
@app.post("/chat")
async def chat(req: ChatRequest):
    user_query = req.message or "Analyze this image"
    image_analysis = None

    if req.image_base64:
        try:
            header, encoded = req.image_base64.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
            with open(file_path, "wb") as f:
                f.write(image_bytes)

            # Analyze the image
            image_analysis = image_analyzer.analyze_fashion_image(file_path, user_query)
        except Exception as e:
            image_analysis = {"error": str(e)}

    # Construct answer (mock or real)
    answer = "Here's what I found based on the image and your message."
    if image_analysis and isinstance(image_analysis, dict) and "raw_analysis" in image_analysis:
        answer = image_analysis["raw_analysis"]

    return {
        "answer": answer,
        "image_analysis": image_analysis
    }
