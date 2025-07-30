import os
import sqlite3
import datetime
import json

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
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    print(f"❌ Error adding column '{col}': {e}")
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

class FashionRecommendationEngine:
    def __init__(self, db: FashionDatabase):
        self.db = db

class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client):
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.client = openai_client

    def handle_image_upload(self, user_id: str, image_path: str, message: str):
        raw_analysis = json.dumps({
            "User's Specific Question": {
                "Shirt Suggestion": "Try a loose pastel or linen shirt to match the jeans."
            },
            "Style Analysis": {
                "Fashion Style": "Streetwear",
                "Vibe": "Relaxed, casual"
            }
        }, indent=2)

        analysis_result = {
            "raw_analysis": raw_analysis,
            "clothing_items": "Available in raw_analysis",
            "colors": "Available in raw_analysis",
            "style_analysis": "Available in raw_analysis"
        }

        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_behavior (user_id, message, image_id, action_type, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    message,
                    os.path.basename(image_path),
                    "image_upload",
                    datetime.datetime.utcnow().isoformat()
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ DB log error: {e}")

        return analysis_result

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw_json = image_analysis["raw_analysis"]
                clean_json = raw_json.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                elif clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

                parsed = json.loads(clean_json)
                suggestion = (
                    parsed.get("User's Specific Question", {}).get("Shirt Suggestion") or
                    parsed.get("Suggested Shirt", {}).get("Details") or
                    parsed.get("Suggested Shirt") or
                    "Try pairing with a crisp white shirt or graphic tee!"
                )
                return {"user_id": user_id, "reply": suggestion, "image_analysis": parsed}
            except Exception as e:
                return {"user_id": user_id, "reply": f"⚠️ Error parsing image data: {e}", "image_analysis": image_analysis}

        return {"user_id": user_id, "reply": "🤔 I don't know how to respond to that.", "image_analysis": image_analysis or {}}
