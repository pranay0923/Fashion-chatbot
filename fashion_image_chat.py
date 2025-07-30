# fashion_image_chat.py

import os
import json
import sqlite3
import base64
from datetime import datetime
from PIL import Image
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# === ENV SETUP ===
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment.")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
openai_client = openai.OpenAI(api_key=OPENAI_KEY)

# === DATABASE ===
class FashionDatabase:
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                product_id INTEGER,
                image_id INTEGER,
                query TEXT,
                preferences TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def track_user_behavior(self, user_id, action_type, product_id=None, image_id=None, query=None, preferences=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_behavior (user_id, action_type, product_id, image_id, query, preferences)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, action_type, product_id, image_id, query, json.dumps(preferences) if preferences else None))
        conn.commit()
        conn.close()

# === IMAGE ANALYZER ===
class ImageAnalyzer:
    def __init__(self, client):
        self.client = client

    def analyze(self, image_path, prompt):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }],
            max_tokens=1000
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"raw_response": response.choices[0].message.content}

# === MAIN CHATBOT ===
class EnhancedFashionChatbot:
    def __init__(self):
        self.db = FashionDatabase()
        self.image_analyzer = ImageAnalyzer(openai_client)
        self.model = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

    def handle_image_upload(self, user_id, image_path, message):
        analysis = self.image_analyzer.analyze(image_path, message)
        self.db.track_user_behavior(user_id, "image_upload", query=message)
        return analysis

    def chat_with_image_context(self, user_id, message, image_analysis=None):
        self.db.track_user_behavior(user_id, "query", query=message)
        context = f"{message}\n\nImage Analysis:\n{json.dumps(image_analysis, indent=2)}" if image_analysis else message
        response = self.model.invoke(context)
        return {"answer": response.content, "context": image_analysis}
