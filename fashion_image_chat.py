import os
import sqlite3
import datetime
import json
import base64

from openai import OpenAI

class FashionDatabase:
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
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
            INSERT INTO user_behavior (user_id, action_type, product_id, image_id, query, preferences, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, action_type, product_id, image_id, query,
            json.dumps(preferences) if preferences else None,
            datetime.datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()

class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client):
        self.db = db
        self.client = openai_client

    def handle_image_upload(self, user_id: str, image_path: str, message: str):
        with open(image_path, "rb") as f:
            img_data = f.read()
        b64_img = base64.b64encode(img_data).decode()

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Analyze this fashion image and provide detailed advice. {message}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }
                ],
                max_tokens=900,
            )
            raw_analysis = response.choices[0].message.content.strip()
        except Exception as e:
            raw_analysis = f"Image analysis failed: {e}"

        analysis_result = {
            "raw_analysis": raw_analysis,
            "clothing_items": "See raw_analysis",
            "colors": "See raw_analysis",
            "style_analysis": "See raw_analysis"
        }
        self.db.track_user_behavior(user_id, "image_upload", image_id=os.path.basename(image_path), query=message)
        return analysis_result

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw = image_analysis["raw_analysis"].strip()
                # Properly quoted strings, no unterminated literals:
                if raw.startswith("```
                    clean = raw[7:]
                elif raw.startswith("```"):
                    clean = raw[3:]
                else:
                    clean = raw
                if clean.endswith("```
                    clean = clean[:-3]
                clean = clean.strip()

                try:
                    parsed = json.loads(clean)
                    suggestion = (
                        parsed.get("Clothing Items") or
                        parsed.get("User's Specific Question", {}).get("Shirt Suggestion") or
                        parsed.get("Suggested Shirt") or
                        clean
                    )
                except Exception:
                    parsed = {}
                    suggestion = clean or "Try a crisp white shirt or a pastel tee!"

                return {
                    "user_id": user_id,
                    "answer": suggestion,
                    "image_analysis": parsed or clean
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "answer": image_analysis["raw_analysis"],
                    "image_analysis": image_analysis
                }
        return {
            "user_id": user_id,
            "answer": "🤔 I don't know how to respond to that.",
            "image_analysis": image_analysis or {}
        }
