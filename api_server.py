# api_server.py

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler

import os
import uvicorn
import tempfile
import traceback

# Import the backend logic factory function only (all other setup is handled within it)
from fashion_image_chat import get_fashion_chatbot

app = FastAPI(title="Fashion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production: set only your frontend domain!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None  # Will be initialized on startup


# --- Exception handlers for robust logging ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return PlainTextResponse(f"Internal Server Error: {str(exc)}", status_code=500)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return await http_exception_handler(request, exc)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return await http_exception_handler(request, exc)

# --- Startup event: create and hold a single chatbot instance ---
@app.on_event("startup")
def startup_event():
    global chatbot
    print("🚀 Initializing backend chatbot...")
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is missing.")
        chatbot = get_fashion_chatbot(openai_key)
        print("✅ Chatbot is ready!")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Startup failed: {e}")
        raise e

@app.get("/")
async def root():
    return {"status": "ok"}

# --- Main backend chat endpoint ---
@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile = File(None),
):
    try:
        print(f"📩 Request: user_id={user_id} message='{message}' image_uploaded={image is not None}")
        response = None
        image_temp_path = None

        # If there is an image, save it (temporarily) and analyze it
        if image:
            try:
                image_content = await image.read()
                if image_content:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(image_content)
                        tmp_file.flush()
                        image_temp_path = tmp_file.name
                        print(f"🖼️ Temp image saved: {image_temp_path}")
            except Exception as read_err:
                print(f"❌ Error reading image file: {read_err}")
                return JSONResponse({"answer": f"Could not read image: {str(read_err)}"}, status_code=400)

        # Process query accordingly
        if image_temp_path:
            # Handle the image and attach its analysis result in the conversation chain
            analysis_result = chatbot.handle_image_upload(user_id, image_temp_path, message)
            try:
                os.remove(image_temp_path)
            except Exception as cleanup_err:
                print(f"⚠️ Warning: Could not clean up temp image: {cleanup_err}")

            if not analysis_result.get("success", False):
                print(f"⚠️ Image analysis failed: {analysis_result.get('error')}")
                return JSONResponse({"answer": f"Image analysis failed: {analysis_result.get('error', 'unknown error')}"}, status_code=500)

            # Now do the main chat (with image context)
            result = chatbot.chat_with_image_context(
                user_id, message, image_analysis=analysis_result.get("analysis", {})
            )
        else:
            # Text query only
            result = chatbot.chat_with_image_context(user_id, message)

        # Format errors for frontend handling (so the UI displays them well)
        if "error" in result:
            return JSONResponse({"answer": f"Error: {result.get('error')}"}, status_code=500)

        # Only send back "answer" (main assistant reply), rest is optional context for frontend
        response_json = {
            "answer": result.get("answer", "🤔 I don't know how to respond to that."),
            "recommendations": result.get("recommendations", []),
            "image_analysis": result.get("image_analysis", {}),
            "user_preferences": result.get("user_preferences", {}),
            # You can add more fields if your frontend wants them
        }
        return JSONResponse(response_json)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"answer": f"Server error: {str(e)}"}, status_code=500)

# --- Run locally for development ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
