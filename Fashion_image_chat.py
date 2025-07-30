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
        """Create/modify DB and add missing columns if needed."""
        # Make sure table exists first (create if needed)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
                -- columns image_id, message, action_type will be added if missing
            )
        """)

        # Add missing columns safely
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
                if "duplicate column name" in str(e).lower():
                    print(f"ℹ️ Column '{col}' already exists.")
                else:
                    print(f"❌ Error adding column '{col}':", e)

        conn.commit()
        conn.close()

    def get_all_products(self):
        # Return 14 fields per product - stub data for demo
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

# === Recommendation Engine (Stub, can be expanded) ===
class FashionRecommendationEngine:
    def __init__(self, db: FashionDatabase):
        self.db = db
    # Add your recommendation logic here...

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
        Process uploaded image:
         - Simulate AI analysis of image content,
         - Log user behavior in DB,
         - Return structured analysis for further chat context.
        """
        # Log upload event
        print(f"📸 Image saved to: uploads/{os.path.basename(image_path)}")
        print("🔍 Analyzing image with AI...")

        # Simulated image analysis (replace with real AI call if available)
        raw_analysis = '''{
          "User's Specific Question": {
            "Shirt Suggestion": "Opt for a loose, oversized white button-up linen shirt to complement the jeans for a relaxed vibe."
          },
          "Style Analysis": {
            "Fashion Style": "Casual streetwear",
            "Vibe": "Youthful and relaxed"
          }
        }'''

        analysis_result = {
            "raw_analysis": raw_analysis,
            "clothing_items": "Analysis available in raw_analysis",
            "colors": "Analysis available in raw_analysis",
            "style_analysis": "Analysis available in raw_analysis"
        }

        # Persist log to DB
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
            print("✅ User behavior logged.")
        except Exception as e:
            print(f"❌ Error processing image: {e}")

        return analysis_result

    def chat_with_image_context(self, user_id: str, message: str, image_analysis=None):
        """
        Generate response based on image analysis and/or text input.
        - Parses raw_analysis JSON string,
        - Extracts relevant reply suggestions,
        - Falls back to default if no usable data.
        """
        if image_analysis and "raw_analysis" in image_analysis:
            try:
                raw_json = image_analysis["raw_analysis"]
                # Clean JSON string of markdown/code block if present
                if raw_json.startswith("```
                    raw_json = raw_json[7:]
                raw_json = raw_json.strip("`\n ")

                parsed = json.loads(raw_json)

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

        # Fallback answer when no valid image analysis is present
        return {
            "user_id": user_id,
            "reply": "🤔 I don't know how to respond to that.",
            "image_analysis": image_analysis or {}
        }
