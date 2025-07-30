from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import os

# If you're using OpenAI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in Render environment

app = FastAPI()

# Allow CORS from any origin (adjust if needed for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request/Response Schemas ----

class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_base64: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    image_analysis: Optional[Dict[str, Any]] = None

# ---- Route ----

@app.post("/chat", response_model=ChatResponse)
async def chat_with_fashion_bot(request: ChatRequest):
    user_id = request.user_id
    message = request.message
    image_data = request.image_base64

    # Step 1: Handle message
    answer = generate_text_response(message)

    # Step 2: Optional image analysis
    image_analysis = None
    if image_data:
        image_analysis = analyze_image_base64(image_data)

    return ChatResponse(
        answer=answer,
        image_analysis=image_analysis
    )

# ---- Helper Functions ----

def generate_text_response(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a fashion assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to generate response: {str(e)}"

def analyze_image_base64(image_data: str) -> dict:
    try:
        # Decode base64 string
        header, base64_str = image_data.split(",", 1)
        image_bytes = base64.b64decode(base64_str)

        # For demo: return fake analysis (replace with real logic later)
        return {
            "dominant_color": "beige",
            "detected_style": "casual",
            "season": "summer"
        }

    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}"}
