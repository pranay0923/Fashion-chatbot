from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64, os
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your deployed Streamlit app domain for production
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

    answer = generate_text_response(message, image_data)
    image_analysis = None
    if image_data:
        image_analysis = analyze_image_base64(image_data)
    return ChatResponse(answer=answer, image_analysis=image_analysis)

def generate_text_response(prompt: str, image_base64: Optional[str] = None) -> str:
    """Call OpenAI for chat response (optionally include image analysis call pipeline)."""
    try:
        final_prompt = prompt
        if image_base64:
            analysis = analyze_image_base64(image_base64)
            if isinstance(analysis, dict):
                final_prompt += f"\n\nImage Analysis: {analysis}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful, trendy AI fashion assistant for all things style and appearance."},
                {"role": "user", "content": final_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to generate response: {str(e)}"

def analyze_image_base64(image_data: str) -> dict:
    try:
        # Decode base64 image, simulate or insert Vision API here for real deployment
        header, base64_str = image_data.split(",", 1)
        image_bytes = base64.b64decode(base64_str)

        # Real image analysis code would be placed here using OpenAI Vision or similar
        # Below is a simulated response. Replace this as needed!
        return {
            "dominant_color": "beige",
            "detected_style": "casual",
            "season": "summer"
        }
    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}"}
