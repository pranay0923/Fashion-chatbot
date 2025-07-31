from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64, os
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_base64: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    image_analysis: Optional[Dict[str, Any]] = None

@app.get("/")
def root():
    return {"message": "Fashion Chatbot API is running. Use POST /chat to interact."}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_fashion_bot(request: ChatRequest):
    user_id = request.user_id
    message = request.message
    image_data = request.image_base64

    answer = generate_text_response(message)
    image_analysis = None
    if image_data:
        image_analysis = analyze_image_base64(image_data)
    return ChatResponse(answer=answer, image_analysis=image_analysis)

def generate_text_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful and trendy fashion assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to generate response: {str(e)}"

def analyze_image_base64(image_data: str) -> dict:
    try:
        header, base64_str = image_data.split(",", 1)
        image_bytes = base64.b64decode(base64_str)
        # Placeholder: Hook in OpenAI Vision API or your model here for richer output.
        return {
            "dominant_color": "beige",
            "detected_style": "casual",
            "season": "summer"
        }
    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}"}
