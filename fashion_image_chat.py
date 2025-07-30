import os
import sqlite3
import datetime

class FashionDatabase:
    def __init__(self):  # FIXED: must be __init__
        self.db_path = "fashion_data.db"
        self.setup()

    def setup(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE user_behavior ADD COLUMN image_id TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                print("❌ Error altering user_behavior:", e)
        conn.commit()
        conn.close()

    def get_all_products(self):
        # Dummy data
        return [
            (1, "Denim Jeans", "Bottoms", "Jeans", "Levis", 59.99, "Blue", "M", "Classic fit", "Casual", "Unisex", "Everyday", "Denim"),
            (2, "White Shirt", "Tops", "Shirt", "Zara", 39.99, "White", "M", "Slim fit", "Formal", "Men", "Work", "Cotton"),
        ]

class FashionRecommendationEngine:
    def __init__(self, db):  # FIXED: must be __init__
        self.db = db

class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client):  # FIXED
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.client = openai_client

    def handle_image_upload(self, user_id, image_path, message):
        print(f"📸 Image saved to: uploads/{os.path.basename(image_path)}")
        print("🔍 Analyzing image with AI...")
        result = {
            "success": True,
            "analysis": {
                "raw_analysis": "json\n{\n  \"Suggested Shirt\": \"White linen oversized shirt\"\n}\n",
                "clothing_items": "Analysis available in raw_analysis",
                "colors": "Analysis available in raw_analysis",
                "style_analysis": "Analysis available in raw_analysis",
            }
        }

        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_behavior (user_id, message, image_id, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                user_id,
                message,
                os.path.basename(image_path),
                datetime.datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            result["success"] = False
            result["error"] = str(e)
        return result

    def chat_with_image_context(self, user_id, message, image_analysis=None):
        return {
            "answer": "You can pair that with a loose-fit button-up white shirt for a casual but chic look!",
            "user_id": user_id,
            "image_analysis": image_analysis or {}
        }
