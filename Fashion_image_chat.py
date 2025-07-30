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
        """Create table if not exists and safely add missing columns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
                -- additional columns below will be added as needed
            )
        """)

        # Add missing columns if they don't exist
        columns = [
            ("image_id", "TEXT"),
            ("message", "TEXT"),
            ("action_type", "TEXT NOT NULL DEFAULT 'unknown'")
        ]

        # Safely add columns, ignore if already present
        for col, col_type in columns:
            try:
                cursor.execute(f"ALTER TABLE user_behavior ADD COLUMN {col} {col_type}")
                print(f"✅ Added column '{col}' to user_behavior table.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"ℹ️ Column '{col}' already exists.")
                else:
                    print(f"❌ Error adding column '{col}': {e}")

        conn.commit()
        conn.close()

    def get_all_products(self):
        """Return list of product tuples with 14 fields each (stub data)."""
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
    # Add recommendation logic here when ready

# === Enhanced Chatbot ===
class EnhancedFashionChatbot:
    def __init__(self, chat_model, retriever, db, rec_engine, openai_client):
        self.chat_model = chat_model
        self.retriever = retriever
        self.db = db
        self.rec_engine = rec_engine
        self.client = openai_client

    def handle_image_upload(self, user_id: str, image_path: str, message: str):
        """
        Simulated image upload handling:
        - Print saved image info,
        - Simulate AI image analysis (replace with real AI vision service),
        - Log to DB,
        - Return analysis dictionary expected by chat_with_image_context.
        """
        print(f"📸 Image saved to: uploads/{os.path.basename(image_path)}")
        print("🔍 Analyzing image with AI...")

        # Simulated raw JSON analysis as a formatted string (no markdown fences)
        raw_analysis = json.dumps({
            "User's Specific Question": {
                "Shirt Suggestion": "Opt for a loose, oversized white button-up linen shirt to complement the jeans for a relaxed vibe."
            },
            "Style Analysis": {
                "Fashion Style": "Casual streetwear",
                "Vibe": "Youthful and relaxed"
            }
        }, indent=2)

        analysis_result = {
            "raw_analysis": raw_analysis,
            "clothing_items": "Analysis available in raw_analysis",
            "colors": "Analysis available in raw_analysis",
            "style_analysis": "Analysis available in raw_analysis"
        }

        # Log the image upload and message to the user_behavior table
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
                    datetime.datetime.utcnow().isoformat()
                )
            )
            conn.commit()
            conn.close()
            print("✅ User behavior logged successfully.")
        except Exception as e:
            print(f"❌ DB logging error: {e}")

        return analysis_result

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        """
        Analyze passed image context JSON, extract meaningful reply.
        Fallback to a generic message when no valid analysis is available.
        """
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw_json = image_analysis["raw_analysis"]
                # Clean whitespace for safe JSON parsing
                clean_json = raw_json.strip()

                # In case markdown fences are included (check and remove)
                if clean_json.startswith("```
                    clean_json = clean_json[7:]
                elif clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```
                    clean_json = clean_json[:-3]

                clean_json = clean_json.strip()

                parsed = json.loads(clean_json)

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
                error_msg = f"Sorry, couldn't parse image analysis. Error: {e}"
                print(f"❌ {error_msg}")
                return {
                    "user_id": user_id,
                    "reply": error_msg,
                    "image_analysis": image_analysis
                }

        # Fallback general reply when no valid image_analysis provided
        return {
            "user_id": user_id,
            "reply": "🤔 I don't know how to respond to that.",
            "image_analysis": image_analysis or {}
        }
