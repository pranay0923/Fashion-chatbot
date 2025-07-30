import os
import json
import sqlite3
import base64
import re
from datetime import datetime

from PIL import Image
from openai import OpenAI

# LangChain imports (adapt imports if library versions differ)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


class FashionDatabase:
    def __init__(self, db_path="fashion.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            subcategory TEXT,
            brand TEXT,
            price REAL,
            color TEXT,
            size TEXT,
            description TEXT,
            style_tags TEXT,
            season TEXT,
            gender TEXT,
            occasion TEXT,
            material TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_behavior (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action_type TEXT,
            product_id INTEGER,
            image_path TEXT,
            query TEXT,
            preferences TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            image_path TEXT,
            description TEXT,
            detected_items TEXT,
            color_analysis TEXT,
            style_analysis TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    # Add methods for adding products, logging behavior, storing images, etc., as needed.
    # For example:

    def add_product(self, product_tuple):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO products (name, category, subcategory, brand, price, color, size, description, style_tags,
                              season, gender, occasion, material, image_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, product_tuple)
        conn.commit()
        conn.close()

    def log_user_behavior(self, user_id, action_type, product_id=None, image_path=None, query=None, preferences=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO user_behavior (user_id, action_type, product_id, image_path, query, preferences)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, action_type, product_id, image_path, query, json.dumps(preferences) if preferences else None))
        conn.commit()
        conn.close()


class ImageAnalysisService:
    def __init__(self, openai_client):
        self.client = openai_client

    def analyze_image(self, image_path, user_query=None):
        """Calls OpenAI image-enabled chat model to analyze uploaded fashion image."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode()

        prompt_text = (
            "Analyze this fashion image and provide detailed info as JSON about: "
            "clothing items, colors, style, season, occasion, and styling suggestions. "
        )
        if user_query:
            prompt_text += f"User question: {user_query}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    }
                ],
                max_tokens=1000,
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            return f"Image analysis failed due to error: {str(e)}"


class FashionRecommendationEngine:
    def __init__(self, db: FashionDatabase, retriever):
        self.db = db
        self.retriever = retriever

    def get_personalized_recommendations(self, query, limit=5):
        # For demo, simple retrieval by query.
        docs = self.retriever.get_relevant_documents(query)
        return docs[:limit]


class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client):
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.image_analysis_service = ImageAnalysisService(openai_client)
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

    def handle_uploaded_image(self, user_id: str, uploaded_file, message):
        filename = f"{user_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
        file_path = os.path.join(self.upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        print(f"Image saved to {file_path}, starting analysis...")
        analysis_text = self.image_analysis_service.analyze_image(file_path, user_query=message)
        print(f"Raw analysis: {analysis_text}")

        # Save analysis results and image record to DB
        # For now, save user image, later parse JSON and store detailed analysis if desired
        self.db.log_user_behavior(user_id, "image_uploaded", image_path=file_path, query=message)
        return {"raw_analysis": analysis_text, "image_path": file_path}

    def chat_with_context(self, user_id: str, message: str, image_analysis=None):
        if image_analysis and "raw_analysis" in image_analysis:
            raw = image_analysis["raw_analysis"]

            # Remove markdown code fences if present
            clean = raw.strip()
            if clean.startswith("```
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```
                clean = clean[:-3]
            clean = clean.strip()

            try:
                parsed_json = json.loads(clean)
                reply = parsed_json.get("clothing_items") or parsed_json.get("suggestions") or json.dumps(parsed_json)
            except Exception:
                reply = raw

            return {"user_id": user_id, "reply": reply, "image_analysis": parsed_json if 'parsed_json' in locals() else raw}
        else:
            # Plain chat without image analysis
            # Implement conversation with context using chat_model and retriever here.
            # For demo, just return simple echo.
            return {"user_id": user_id, "reply": f"You asked: {message}", "image_analysis": {}}

