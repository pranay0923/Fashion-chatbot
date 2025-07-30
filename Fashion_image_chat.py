import os
import sqlite3
import datetime
import json
from typing import Optional
from PIL import Image
import base64

from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# === FashionDatabase ===
class FashionDatabase:
    def __init__(self):
        self.db_path = "fashion_data.db"
        self.setup()

    def setup(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        columns = [
            ("image_id", "TEXT"),
            ("message", "TEXT"),
            ("action_type", "TEXT NOT NULL DEFAULT 'unknown'")
        ]
        for col, col_type in columns:
            try:
                cursor.execute(f"ALTER TABLE user_behavior ADD COLUMN {col} {col_type}")
                print(f"✅ Column '{col}' added.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"ℹ️ Column '{col}' already exists.")
                else:
                    print(f"❌ Error adding column '{col}': {e}")
        conn.commit()
        conn.close()

    def get_all_products(self):
        return [
            (
                1, "Denim Jeans", "Bottoms", "Jeans", "Levis", 59.99, "Blue", "M",
                "High-waisted jeans with button details",
                "Street", "Summer", "Unisex", "Casual", "Denim"
            ),
            (
                2, "White Shirt", "Tops", "Shirt", "Zara", 39.99, "White", "L",
                "Oversized cotton shirt with rolled sleeves",
                "Formal", "All Seasons", "Women", "Work", "Cotton"
            )
        ]


# === FashionRecommendationEngine ===
class FashionRecommendationEngine:
    def __init__(self, db: FashionDatabase):
        self.db = db
    # TODO: Add smart recommendations


# === EnhancedFashionChatbot ===
class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client: OpenAI):
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.client = openai_client

    def handle_image_upload(self, user_id: str, image_path: str, message: str):
        print(f"📸 Saving image: {image_path}")
        print("🔍 Running AI analysis...")

        with open(image_path, "rb") as f:
            img_data = f.read()

        base64_image = base64.b64encode(img_data).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Suggest a shirt for this jeans outfit. {message}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=700
            )
            content = response.choices[0].message.content
            raw_analysis = content.strip()
        except Exception as e:
            print(f"❌ AI image analysis failed: {e}")
            raw_analysis = "Image analysis failed"

        analysis_result = {
            "raw_analysis": raw_analysis,
            "clothing_items": "In raw_analysis",
            "colors": "In raw_analysis",
            "style_analysis": "In raw_analysis"
        }

        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_behavior (user_id, message, image_id, action_type, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                message,
                os.path.basename(image_path),
                "image_upload",
                datetime.datetime.utcnow().isoformat()
            ))
            conn.commit()
            conn.close()
            print("✅ Behavior logged.")
        except Exception as e:
            print(f"❌ DB log error: {e}")

        return analysis_result

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw = image_analysis["raw_analysis"]
                clean = raw.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                elif clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                parsed = json.loads(clean)

                suggestion = (
                    parsed.get("User's Specific Question", {}).get("Shirt Suggestion")
                    or parsed.get("Suggested Shirt", {}).get("Details")
                    or parsed.get("Suggested Shirt")
                    or "Try a clean white button-up or a pastel crop top."
                )
                return {
                    "user_id": user_id,
                    "reply": suggestion,
                    "image_analysis": parsed
                }
            except Exception as e:
                print(f"❌ JSON parse error: {e}")
                return {
                    "user_id": user_id,
                    "reply": f"⚠️ Could not parse analysis: {e}",
                    "image_analysis": image_analysis
                }

        return {
            "user_id": user_id,
            "reply": "🤔 I don't know how to respond to that.",
            "image_analysis": image_analysis or {}
        }
