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

        # ✅ Add 'image_id' column if missing
        try:
            cursor.execute("ALTER TABLE user_behavior ADD COLUMN image_id TEXT")
            print("✅ Column 'image_id' added to user_behavior table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("ℹ️ Column 'image_id' already exists.")
            else:
                print("❌ Error altering table (image_id):", e)

        # ✅ Add 'message' column if missing
        try:
            cursor.execute("ALTER TABLE user_behavior ADD COLUMN message TEXT")
            print("✅ Column 'message' added to user_behavior table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("ℹ️ Column 'message' already exists.")
            else:
                print("❌ Error altering table (message):", e)

        # ✅ Add 'action_type' column if missing
        try:
            cursor.execute("ALTER TABLE user_behavior ADD COLUMN action_type TEXT NOT NULL DEFAULT 'unknown'")
            print("✅ Column 'action_type' added to user_behavior table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("ℹ️ Column 'action_type' already exists.")
            else:
                print("❌ Error altering table (action_type):", e)

        conn.commit()
        conn.close()

    def get_all_products(self):
        # ✅ Return 14 fields per product
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


# === Enhanced Chatbot ===
class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client):
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.client = openai_client

    def handle_image_upload(self, user_id: str, image_path: str, message: str):
        # Simulate analysis
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

        # Log to DB
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO user_behavior (user_id, message, image_id, action_type, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    message,
                    os.path.basename(image_path),
                    "image_upload",
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
        return {
            "user_id": user_id,
            "reply": "You can pair that with a loose-fit white shirt or a pastel oversized tee!",
            "image_analysis": image_analysis or {}
        }
