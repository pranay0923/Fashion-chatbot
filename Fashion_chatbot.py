import os
import sqlite3
import datetime
import json
import base64

from openai import OpenAI


class FashionDatabase:
    """
    Handles all database operations including product management,
    user images, preferences, and user behavior tracking.
    """
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action_type TEXT,
                product_id INTEGER,
                image_id INTEGER,
                query TEXT,
                preferences TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(id),
                FOREIGN KEY (image_id) REFERENCES user_images(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_colors TEXT,
                preferred_brands TEXT,
                preferred_styles TEXT,
                size_preference TEXT,
                budget_range TEXT,
                body_type TEXT,
                style_personality TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def get_all_products(self):
        """
        Returns a list of product tuples.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, category, subcategory, brand, price, color, size, description, style_tags, season, gender, occasion, material FROM products")
        products = cursor.fetchall()
        conn.close()
        return products

    def save_user_image(self, user_id, image_path, description=None, detected_items=None, color_analysis=None, style_analysis=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_images (user_id, image_path, image_description, detected_items, color_analysis, style_analysis)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            image_path,
            description,
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
            INSERT INTO user_behavior (user_id, action_type, product_id, image_id, query, preferences)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            action_type,
            product_id,
            image_id,
            query,
            json.dumps(preferences) if preferences else None
        ))
        conn.commit()
        conn.close()


class FashionRecommendationEngine:
    """
    Stub for a recommendation engine compatible with LangChain Retriever or any logic.
    """
    def __init__(self, db: FashionDatabase):
        self.db = db
    # Implement recommendation logic here if desired


class EnhancedFashionChatbot:
    """
    Processes user inputs, image uploads, analyzes via OpenAI Vision and chat, and handles conversations.
    """
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client: OpenAI):
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.client = openai_client

    def handle_image_upload(self, user_id: str, image_path: str, message: str):
        print(f"📸 Saving image: {image_path}")

        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64_image = base64.b64encode(img_bytes).decode()

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Analyze this fashion image. {message}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        ]
                    }
                ],
                max_tokens=900,
            )
            content = response.choices[0].message.content
            raw_analysis = content.strip()
        except Exception as e:
            print(f"❌ AI analysis failed: {e}")
            raw_analysis = "Image analysis failed."

        analysis = {
            "raw_analysis": raw_analysis,
            "clothing_items": "See raw_analysis",
            "color_analysis": "See raw_analysis",
            "style_analysis": "See raw_analysis",
        }

        try:
            self.db.track_user_behavior(user_id, "image_upload", image_id=os.path.basename(image_path), query=message)
        except Exception as e:
            print(f"❌ DB log error: {e}")

        return analysis

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw = image_analysis["raw_analysis"]
                clean = raw.strip()
                if clean.startswith("```
                    clean = clean[7:]
                elif clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```
                    clean = clean[:-3]
                clean = clean.strip()

                try:
                    parsed = json.loads(clean)
                    suggestion = (
                        parsed.get("clothing_items") or
                        parsed.get("Clothing Items") or
                        parsed.get("User's Specific Question", {}).get("Shirt Suggestion") or
                        parsed.get("Suggested Shirt") or
                        raw
                    )
                except json.JSONDecodeError:
                    parsed = {}
                    suggestion = clean or "Try a crisp white shirt or a pastel tee!"

                return {
                    "user_id": user_id,
                    "reply": suggestion,
                    "image_analysis": parsed
                }
            except Exception as e:
                print(f"❌ JSON parse error: {e}")
                return {
                    "user_id": user_id,
                    "reply": image_analysis.get("raw_analysis", ""),
                    "image_analysis": image_analysis,
                }
        return {
            "user_id": user_id,
            "reply": "🤔 I don't know how to respond to that.",
            "image_analysis": image_analysis or {},
        }
