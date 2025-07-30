import os
import sqlite3
import datetime

# === Database Wrapper ===
class FashionDatabase:
    def __init__(self):
        self.db_path = "fashion_data.db"
        self.setup()

    def setup(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ✅ Add 'image_id' column to user_behavior table if missing
        try:
            cursor.execute("ALTER TABLE user_behavior ADD COLUMN image_id TEXT")
            print("✅ Column 'image_id' added to user_behavior table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("ℹ️ Column 'image_id' already exists.")
            else:
                print("❌ Error altering user_behavior table:", e)

        conn.commit()
        conn.close()

    def get_all_products(self):
        # ✅ Return 14 fields per product (required by api_server.py)
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


# === Recommendation Engine (stub for now) ===
class FashionRecommendationEngine:
    def __init__(self, db: FashionDatabase):
        self.db = db


# === Main Chatbot Class ===
class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client):
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.client = openai_client

    def handle_image_upload(self, user_id: str, image_path: str, message: str):
        # ✅ Simulate AI image analysis
        print(f"📸 Image saved to: uploads/{os.path.basename(image_path)}")
        print("🔍 Analyzing image with AI...")

        analysis_result = {
            "success": True,
            "analysis": {
                "raw_analysis": "```json\n{\n  \"Suggested Shirt\": \"White linen oversized shirt\"\n}\n```",
                "clothing_items": "Analysis available in raw_analysis",
                "colors": "Analysis available in raw_analysis",
                "style_analysis": "Analysis available in raw_analysis",
            }
        }

        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO user_behavior (user_id, message, image_id, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    message,
                    os.path.basename(image_path),
                    datetime.datetime.now().isoformat()
                )
            )

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            analysis_result["success"] = False
            analysis_result["error"] = str(e)

        return analysis_result

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        # ✅ Return a mock response
        return {
            "user_id": user_id,
            "reply": "You can pair that with a loose-fit white shirt or a pastel oversized tee!",
            "image_analysis": image_analysis or {}
        }
