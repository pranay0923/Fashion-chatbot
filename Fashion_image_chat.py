import os
import sqlite3
import datetime
import json

# === Database Wrapper ===
class FashionDatabase:
    def __init__(self):
        self.db_path = "fashion_data.db"
        self.setup()

    def setup(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Add missing columns if needed
        columns = [
            ("image_id", "TEXT"),
            ("message", "TEXT"),
            ("action_type", "TEXT NOT NULL DEFAULT 'unknown'")
        ]

        for col, col_type in columns:
            try:
                cursor.execute(f"ALTER TABLE user_behavior ADD COLUMN {col} {col_type}")
                print(f"✅ Column '{col}' added to user_behavior table.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"ℹ️ Column '{col}' already exists.")
                else:
                    print(f"❌ Error altering table ({col}):", e)

        conn.commit()
        conn.close()

    def get_all_products(self):
        # Return 14 fields per product
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
        print(f"📸 Image saved to: uploads/{os.path.basename(image_path)}")
        print("🔍 Analyzing image with AI...")

        # Simulated image analysis result
        analysis_result = {
            "success": True,
            "analysis": {
                "raw_analysis": '''```json
{
  "User's Specific Question": {
    "Shirt Suggestion": "Opt for a loose, oversized white button-up linen shirt to complement the jeans for a relaxed vibe."
  },
  "Style Analysis": {
    "Fashion Style": "Casual streetwear",
    "Vibe": "Youthful and relaxed"
  }
}
```''',
                "clothing_items": "Analysis available in raw_analysis",
                "colors": "Analysis available in raw_analysis",
                "style_analysis": "Analysis available in raw_analysis"
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
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw_json = image_analysis["raw_analysis"]
                print("🔎 Raw analysis received:", raw_json)

                # Remove formatting wrappers
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                raw_json = raw_json.strip("`\n ")

                parsed = json.loads(raw_json)

                # Try various fields
                suggestion = (
                    parsed.get("User's Specific Question", {}).get("Shirt Suggestion") or
                    parsed.get("Suggested Shirt", {}).get("Details") or
                    parsed.get("Suggested Shirt") or
                    "Try pairing it with a crisp white shirt or a pastel tee!"
                )

                return {
                    "user_id": user_id,
                    "reply": suggestion,
                    "image_analysis": parsed
                }

            except Exception as e:
                print("❌ Error parsing raw_analysis:", e)
                return {
                    "user_id": user_id,
                    "reply": f"Sorry, I couldn't interpret the outfit. Error: {str(e)}",
                    "image_analysis": image_analysis
                }

        return {
            "user_id": user_id,
            "reply": "🤔 I don't know how to respond to that.",
            "image_analysis": image_analysis or {}
        }
