import os
import sqlite3
import datetime
import json
import base64
from openai import OpenAI

# === FashionDatabase ===
class FashionDatabase:
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.setup()

    def setup(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action_type TEXT,
                product_id INTEGER,
                image_id TEXT,
                query TEXT,
                preferences TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_description TEXT,
                detected_items TEXT,
                color_analysis TEXT,
                style_analysis TEXT,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def save_user_image(self, user_id, image_path, description=None, detected_items=None, color_analysis=None, style_analysis=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_images (user_id, image_path, image_description, detected_items, color_analysis, style_analysis)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, image_path, description,
            json.dumps(detected_items) if detected_items else None,
            json.dumps(color_analysis) if color_analysis else None,
            json.dumps(style_analysis) if style_analysis else None
        ))
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return image_id

    def track_user_behavior(self, user_id, action_type, product_id=None, image_id=None, query=None, preferences=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_behavior (user_id, timestamp, action_type, product_id, image_id, query, preferences)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            datetime.datetime.utcnow().isoformat(),
            action_type,
            product_id,
            image_id,
            query,
            json.dumps(preferences) if preferences else None
        ))
        conn.commit()
        conn.close()

    def get_all_products(self):
        return [
            (
                1, "Denim Jeans", "Bottoms", "Jeans", "Levis", 59.99, "Blue", "M",
                "High-waisted straight-leg jeans with button detail",
                "Street", "Summer", "Unisex", "Casual", "Denim"
            ),
            (
                2, "White Shirt", "Tops", "Shirt", "Zara", 39.99, "White", "L",
                "Oversized white cotton shirt with rolled sleeves",
                "Formal", "All Seasons", "Women", "Work", "Cotton"
            )
        ]

# === Recommendation Engine (Stub) ===
class FashionRecommendationEngine:
    def __init__(self, db: FashionDatabase):
        self.db = db

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
                            {"type": "text", "text": f"Analyze this fashion image. {message}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=900
            )
            content = response.choices.message.content
            raw_analysis = content.strip()
        except Exception as e:
            print(f"❌ AI image analysis failed: {e}")
            raw_analysis = "Image analysis failed"

        analysis_result = {
            "raw_analysis": raw_analysis,
            "clothing_items": "See raw_analysis",
            "colors": "See raw_analysis",
            "style_analysis": "See raw_analysis"
        }

        try:
            self.db.track_user_behavior(
                user_id,
                "image_upload",
                image_id=os.path.basename(image_path),
                query=message
            )
        except Exception as e:
            print(f"❌ DB log error: {e}")

        return analysis_result

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw = image_analysis["raw_analysis"]
                clean = raw.strip()
                # Correct triple-backtick fence handling -- ALL QUOTED!
                if clean.startswith("```json"):
                    clean = clean[7:]
                elif clean.startswith("```
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
                try:
                    parsed = json.loads(clean)
                    suggestion = (
                        parsed.get("Clothing Items")
                        or parsed.get("User's Specific Question", {}).get("Shirt Suggestion")
                        or parsed.get("Suggested Shirt")
                        or raw
                    )
                except Exception:
                    parsed = {}
                    suggestion = clean if clean else "Try a crisp white shirt or a pastel tee!"
                return {
                    "user_id": user_id,
                    "reply": suggestion,
                    "image_analysis": parsed or clean
                }
            except Exception as e:
                print(f"❌ JSON parse error: {e}")
                return {
                    "user_id": user_id,
                    "reply": image_analysis["raw_analysis"],
                    "image_analysis": image_analysis
                }
        return {
            "user_id": user_id,
            "reply": "🤔 I don't know how to respond to that.",
            "image_analysis": image_analysis or {}
        }
