# fashion_image_chat.py

import os
import json
import base64
from datetime import datetime
from PIL import Image
import sqlite3
import openai

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ================================
# OpenAI and LangChain setup
# ================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# ================================
# Database Setup
# ================================
class FashionDatabase:
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_images (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                image_base64 TEXT,
                analysis TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def save_image_analysis(self, user_id, image_base64, analysis_dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_images (user_id, image_base64, analysis)
            VALUES (?, ?, ?)
        ''', (user_id, image_base64[:100], json.dumps(analysis_dict)))
        conn.commit()
        conn.close()


# ================================
# Image Analysis Service
# ================================
class ImageAnalysisService:
    def __init__(self, openai_client):
        self.client = openai_client

    def analyze_base64_image(self, base64_str, user_query=None):
        prompt = f"""
        Analyze this fashion image and return details as JSON:
        1. Clothing Items
        2. Dominant Colors
        3. Style Summary
        4. Suitable Occasion
        5. Seasonality
        6. Styling Tips
        7. Similar Products

        User Query: {user_query or "N/A"}
        Format: JSON with relevant keys
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_str}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            raw = response.choices[0].message.content

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw_analysis": raw, "error": "Could not parse JSON"}
        except Exception as e:
            return {"error": str(e), "message": "Vision API call failed"}


# ================================
# Main Handler (Backend Friendly)
# ================================
class FashionImageChatHandler:
    def __init__(self):
        self.db = FashionDatabase()
        self.analyzer = ImageAnalysisService(openai_client)

    def image_file_to_base64(self, file):
        image_bytes = file.read()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def process_image_and_chat(self, user_id, file, user_query=""):
        base64_image = self.image_file_to_base64(file)
        analysis = self.analyzer.analyze_base64_image(base64_image, user_query)

        self.db.save_image_analysis(user_id, base64_image, analysis)

        return {
            "user_id": user_id,
            "analysis": analysis,
        }
