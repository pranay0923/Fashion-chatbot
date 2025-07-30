
# ===== PART 1: IMPORT ALL REQUIRED LIBRARIES =====
import os
import json
import sqlite3
import base64
from datetime import datetime
from PIL import Image
import io
import requests
from getpass import getpass
import openai  # Added this import

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# In[354]:


# ===== PART 2: OPENAI API SETUP =====
from getpass import getpass
OPENAI_KEY = getpass("Enter your API Token here: ")
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

# Initialize OpenAI models
chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize native OpenAI client for image analysis
openai_native_client = openai.OpenAI(api_key=OPENAI_KEY)


# In[356]:


# ===== PART 3: ENHANCED DATABASE CLASS =====
class FashionDatabase:
    """Enhanced database class with image support"""

    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Fashion products table (original)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
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
        ''')

        # NEW: User uploaded images table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_images (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_description TEXT,
            detected_items TEXT,
            color_analysis TEXT,
            style_analysis TEXT,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Enhanced user behavior tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behavior (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            product_id INTEGER,
            image_id INTEGER,
            query TEXT,
            preferences TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (id),
            FOREIGN KEY (image_id) REFERENCES user_images (id)
        )
        ''')

        # User preferences table
        cursor.execute('''
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
        ''')

        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")

    def add_product(self, **product_data):
        """Add a new fashion product to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ', '.join(['?' for _ in product_data])
        columns = ', '.join(product_data.keys())
        sql = f"INSERT INTO products ({columns}) VALUES ({placeholders})"

        cursor.execute(sql, list(product_data.values()))
        product_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return product_id

    def get_all_products(self):
        """Get all products from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products")
        products = cursor.fetchall()
        conn.close()
        return products

    def save_user_image(self, user_id, image_path, description=None, detected_items=None, 
                       color_analysis=None, style_analysis=None):
        """Save user uploaded image with analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO user_images (user_id, image_path, image_description, 
                                detected_items, color_analysis, style_analysis)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, image_path, description, 
              json.dumps(detected_items) if detected_items else None,
              json.dumps(color_analysis) if color_analysis else None,
              json.dumps(style_analysis) if style_analysis else None))
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return image_id

    def get_user_images(self, user_id, limit=10):
        """Get user's uploaded images"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT * FROM user_images 
        WHERE user_id = ? 
        ORDER BY upload_timestamp DESC 
        LIMIT ?
        ''', (user_id, limit))
        result = cursor.fetchall()
        conn.close()
        return result

    def track_user_behavior(self, user_id, action_type, product_id=None, image_id=None, query=None, preferences=None):
        """Track user behavior for recommendations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO user_behavior (user_id, action_type, product_id, image_id, query, preferences)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, action_type, product_id, image_id, query, 
              json.dumps(preferences) if preferences else None))
        conn.commit()
        conn.close()

    def update_user_preferences(self, user_id, preferences):
        """Update user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO user_preferences 
        (user_id, preferred_colors, preferred_brands, preferred_styles, 
         size_preference, budget_range, body_type, style_personality)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, 
              json.dumps(preferences.get('colors', [])),
              json.dumps(preferences.get('brands', [])),
              json.dumps(preferences.get('styles', [])),
              preferences.get('size', ''),
              preferences.get('budget', ''),
              preferences.get('body_type', ''),
              preferences.get('style_personality', '')))
        conn.commit()
        conn.close()


# In[358]:


# ===== PART 4: IMAGE ANALYSIS SERVICE (FIXED) =====
class ImageAnalysisService:
    """Service for analyzing fashion images using AI"""

    def __init__(self, openai_native_client):
        self.openai_client = openai_native_client  # Use native OpenAI client

    def analyze_fashion_image(self, image_path_or_base64, user_query=None):
        """Analyze fashion items in uploaded image using GPT-4 Vision"""
        try:
            # Prepare the image data
            if image_path_or_base64.startswith('data:image'):
                # Already base64 encoded
                image_data = image_path_or_base64
            else:
                # Convert file path to base64
                with open(image_path_or_base64, "rb") as image_file:
                    image_content = base64.b64encode(image_file.read()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{image_content}"

            # Create detailed analysis prompt
            analysis_prompt = f"""
            Analyze this fashion image and provide detailed information in the following categories:

            1. **Clothing Items**: List all visible clothing items with detailed descriptions
            2. **Colors**: Identify dominant colors and overall color palette
            3. **Style Analysis**: Describe the fashion style, aesthetic, and overall vibe
            4. **Occasion**: What occasions would this outfit be suitable for
            5. **Season**: What season is this outfit appropriate for
            6. **Body Type**: What body types would this outfit flatter
            7. **Styling Tips**: Specific advice on how to style or improve this look
            8. **Similar Items**: Suggest what similar items to look for

            User's specific question: {user_query if user_query else "General fashion analysis"}

            Please provide a detailed response in JSON format with the above categories as keys.
            """

            # Call OpenAI GPT-4 Vision API using the correct native client
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Using gpt-4o which supports vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ]
                    }
                ],
                max_tokens=1000
            )

            # Parse the response
            analysis_text = response.choices[0].message.content

            # Try to parse as JSON, fallback to raw text if fails
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, structure the response
                analysis_json = {
                    "raw_analysis": analysis_text,
                    "clothing_items": "Analysis available in raw_analysis",
                    "colors": "Analysis available in raw_analysis",
                    "style_analysis": "Analysis available in raw_analysis"
                }

            return analysis_json

        except Exception as e:
            return {
                "error": str(e), 
                "message": "Failed to analyze image. Please check your image file and API key."
            }

    def extract_colors_from_image(self, image_path):
        """Extract dominant colors from image using PIL"""
        try:
            # Open and process the image
            image = Image.open(image_path)
            image = image.convert('RGB')

            # Get dominant colors
            colors = image.getcolors(maxcolors=256*256*256)
            if colors:
                # Sort by frequency and get top 5 colors
                dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
                color_info = []

                for count, rgb in dominant_colors:
                    color_info.append({
                        "rgb": rgb,
                        "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                        "frequency": count,
                        "percentage": round((count / sum([c[0] for c in colors])) * 100, 2)
                    })

                return color_info

        except Exception as e:
            return {"error": str(e)}

        return []


# In[359]:


# ===== PART 5: RECOMMENDATION ENGINE (FIXED) =====
class FashionRecommendationEngine:
    """Enhanced recommendation engine for fashion products"""

    def __init__(self, fashion_db):
        self.fashion_db = fashion_db

    def get_personalized_recommendations(self, user_id, query, limit=5):
        """Get personalized product recommendations"""
        try:
            products = self.fashion_db.get_all_products()
            if not products:
                return []

            # Convert products to recommendation format
            recommendations = []
            for product in products[:limit]:
                rec = {
                    'product_id': product[0],
                    'name': product[1],
                    'category': product[2] if len(product) > 2 else 'Fashion Item',
                    'brand': product[4] if len(product) > 4 else 'Unknown Brand',
                    'price': product[5] if len(product) > 5 else 0,
                    'color': product[6] if len(product) > 6 else 'Various',
                    'description': product[8] if len(product) > 8 else 'Stylish fashion item'
                }
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def analyze_user_preferences(self, user_id):
        """Analyze user preferences from behavior"""
        # Simple preference analysis - can be enhanced
        return {
            'preferred_colors': ['blue', 'black', 'white'],
            'preferred_brands': ['Nike', 'Adidas'],
            'preferred_styles': ['casual', 'sporty'],
            'budget_range': 'medium'
        }


# In[362]:


# ===== PART 6: SAMPLE DATA LOADING (FIXED) =====
def load_sample_fashion_data(fashion_db):
    """Load sample fashion data for testing"""

    sample_products = [
        {
            'name': 'Classic Blue Jeans',
            'category': 'Bottoms',
            'subcategory': 'Jeans',
            'brand': 'Levi\'s',  # Fixed HTML entity
            'price': 79.99,
            'color': 'Blue',
            'size': 'M',
            'description': 'Classic straight-leg blue jeans perfect for casual wear',
            'style_tags': 'casual,classic,versatile',
            'season': 'all-season',
            'gender': 'unisex',
            'occasion': 'casual,everyday',
            'material': 'cotton,denim'
        },
        {
            'name': 'White Cotton T-Shirt',
            'category': 'Tops',
            'subcategory': 'T-Shirts',
            'brand': 'Gap',
            'price': 19.99,
            'color': 'White',
            'size': 'M',
            'description': 'Comfortable white cotton t-shirt for everyday wear',
            'style_tags': 'basic,casual,comfortable',
            'season': 'all-season',
            'gender': 'unisex',
            'occasion': 'casual,everyday',
            'material': 'cotton'
        },
        {
            'name': 'Black Leather Jacket',
            'category': 'Outerwear',
            'subcategory': 'Jackets',
            'brand': 'Zara',
            'price': 199.99,
            'color': 'Black',
            'size': 'M',
            'description': 'Stylish black leather jacket for edgy looks',
            'style_tags': 'edgy,rock,cool',
            'season': 'fall,winter',
            'gender': 'unisex',
            'occasion': 'casual,party,date',
            'material': 'leather'
        },
        {
            'name': 'Summer Floral Dress',
            'category': 'Dresses',
            'subcategory': 'Casual Dresses',
            'brand': 'H&M',  # Fixed HTML entity
            'price': 49.99,
            'color': 'Floral',
            'size': 'M',
            'description': 'Light and airy floral dress perfect for summer',
            'style_tags': 'feminine,floral,light',
            'season': 'spring,summer',
            'gender': 'women',
            'occasion': 'casual,date,brunch',
            'material': 'polyester,viscose'
        },
        {
            'name': 'Running Sneakers',
            'category': 'Footwear',
            'subcategory': 'Sneakers',
            'brand': 'Nike',
            'price': 129.99,
            'color': 'White/Blue',
            'size': '9',
            'description': 'Comfortable running sneakers with great support',
            'style_tags': 'sporty,athletic,comfortable',
            'season': 'all-season',
            'gender': 'unisex',
            'occasion': 'sport,casual',
            'material': 'synthetic,mesh'
        }
    ]

    # Add sample products to database
    for product in sample_products:
        fashion_db.add_product(**product)

    print(f"✅ Loaded {len(sample_products)} sample products into database!")
    return sample_products


# In[364]:


# ===== PART 7: VECTOR STORE SETUP (FAISS VERSION) =====
from langchain_community.vectorstores import FAISS

def create_fashion_vector_store(fashion_db, embeddings_model):
    """Create vector store from fashion database using FAISS"""

    # Get all products from database
    products = fashion_db.get_all_products()

    if not products:
        print("No products found. Loading sample data...")
        load_sample_fashion_data(fashion_db)
        products = fashion_db.get_all_products()

    # Convert products to documents for vector store
    documents = []
    for product in products:
        # Create document text with all product information
        doc_text = f"""
        Name: {product[1]}
        Category: {product[2]} - {product[3]}
        Brand: {product[4]}
        Price: ${product[5]}
        Color: {product[6]}
        Size: {product[7]}
        Description: {product[8]}
        Style: {product[9]}
        Season: {product[10]}
        Gender: {product[11]}
        Occasion: {product[12]}
        Material: {product[13]}
        """

        # Create document with metadata
        from langchain.docstore.document import Document
        doc = Document(
            page_content=doc_text.strip(),
            metadata={
                'product_id': product[0],
                'name': product[1],
                'category': product[2],
                'brand': product[4],
                'price': product[5],
                'color': product[6]
            }
        )
        documents.append(doc)

    # Create vector store using FAISS (more stable than ChromaDB)
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings_model
    )

    # Save the vector store locally
    vectorstore.save_local("fashion_faiss_db")

    print(f"✅ Created FAISS vector store with {len(documents)} fashion products!")
    return vectorstore


# In[366]:


# ===== PART 8: ENHANCED FASHION CHATBOT (FIXED) =====
import re

class EnhancedFashionChatbot:
    """Enhanced Fashion Chatbot with image upload and analysis capabilities"""

    def __init__(self, chatgpt, retriever, fashion_db, rec_engine, openai_native_client):
        self.chatgpt = chatgpt
        self.retriever = retriever
        self.fashion_db = fashion_db
        self.rec_engine = rec_engine
        # FIXED: Use native OpenAI client for image analysis
        self.image_analyzer = ImageAnalysisService(openai_native_client)
        self.setup_chains()

        # Create uploads directory for images
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
        print(f"✅ Enhanced Fashion Chatbot initialized!")
        print(f"📁 Upload directory created: {self.upload_dir}")

    def setup_chains(self):
        """Setup LangChain conversation chains"""

        # Enhanced rephrase prompt for image context
        rephrase_system_prompt = """
        You are a fashion stylist assistant with image analysis capabilities. 
        Given a conversation history and the latest user question (which might include 
        image analysis results), formulate a standalone question that captures the user's 
        fashion needs, style preferences, and visual context from uploaded images.

        Consider factors like: style preference, body type, occasion, budget, color preferences, 
        brand preferences, seasonal needs, and visual elements from uploaded images.
        """

        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", rephrase_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.history_aware_retriever = create_history_aware_retriever(
            self.chatgpt, self.retriever, rephrase_prompt
        )

        # Enhanced QA prompt with image context
        qa_system_prompt = """
        You are an expert fashion stylist and personal shopping assistant with image analysis capabilities.

        Your expertise includes:
        - Fashion trends and seasonal styles
        - Body types and flattering fits
        - Color theory and coordination
        - Brand knowledge and quality assessment
        - Occasion-appropriate dressing
        - Budget-conscious styling
        - Visual analysis of uploaded fashion images

        When users upload images:
        1. Analyze the visual elements thoroughly
        2. Provide specific feedback on the items shown
        3. Suggest improvements or alternatives
        4. Recommend similar products from the database
        5. Give styling advice based on what you see

        For each recommendation:
        1. Explain WHY it suits the user (including visual analysis if image provided)
        2. Suggest styling tips
        3. Mention care instructions if relevant
        4. Provide alternatives in different price ranges
        5. Reference visual elements from uploaded images when relevant

        Always be friendly, helpful, and fashion-forward in your responses.

        Context: {context}
        Image Analysis Results: {image_analysis}
        """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.chatgpt, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

        # Memory management with SQLite
        def get_session_history(session_id):
            return SQLChatMessageHistory(session_id, "sqlite:///fashion_memory.db")

        self.conversational_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        print("✅ Conversation chains setup complete!")

    def handle_image_upload(self, user_id, image_file_path, query=None):
        """Handle image upload and analysis"""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}_{timestamp}.jpg"
            image_path = os.path.join(self.upload_dir, filename)

            # Copy/save the image file
            if os.path.exists(image_file_path):
                with open(image_file_path, 'rb') as src, open(image_path, 'wb') as dst:
                    dst.write(src.read())
            else:
                return {"success": False, "error": "Image file not found"}

            print(f"📸 Image saved to: {image_path}")

            # Analyze the image using AI
            print("🔍 Analyzing image with AI...")
            image_analysis = self.image_analyzer.analyze_fashion_image(image_path, query)
            print("DEBUG_IMAGE_ANALYSIS_RESULT:", image_analysis)

            # Extract colors from image
            print("🎨 Extracting color palette...")
            color_analysis = self.image_analyzer.extract_colors_from_image(image_path)

            # Save to database
            image_id = self.fashion_db.save_user_image(
                user_id=user_id,
                image_path=image_path,
                description=query,
                detected_items=image_analysis.get('clothing_items') if isinstance(image_analysis, dict) else None,
                color_analysis=color_analysis,
                style_analysis=image_analysis.get('style_analysis') if isinstance(image_analysis, dict) else None
            )

            # Track user behavior
            self.fashion_db.track_user_behavior(
                user_id, "image_upload", image_id=image_id, query=query
            )

            print("✅ Image analysis complete!")

            return {
                "success": True,
                "image_id": image_id,
                "image_path": image_path,
                "analysis": image_analysis,
                "colors": color_analysis
            }
        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            return {"success": False, "error": str(e)}

    def chat_with_image_context(self, user_id, message, image_analysis=None):
        """Enhanced chat method with up-to-date image context. Recommends ONLY IF requested."""
        # Track user query
        self.fashion_db.track_user_behavior(user_id, "query", query=message)

        # Detect if user is asking for recommendations
        rec_keywords = [
            r"\brecommend\b",
            r"\bsuggest\b",
            r"\bshow me\b",
            r"\boptions\b",
            r"\bcan you give me\b",
            r"\bwhat should I buy\b",
            r"\bgive me choices\b",
            r"\bany (good )?(options|suggestions|recommendations)\b"
        ]
        rec_query = any(re.search(pattern, message, re.IGNORECASE) for pattern in rec_keywords)

        # Only generate recommendations if user asks for them
        if rec_query:
            recommendations = self.rec_engine.get_personalized_recommendations(
                user_id, message, limit=3
            )
        else:
            recommendations = []

        # Prepare input with fresh image context
        enhanced_input = message
        if image_analysis:
            # Display analysis if no error, otherwise handle gracefully
            if isinstance(image_analysis, dict) and image_analysis.get("error"):
                enhanced_input += f"\n\n[Image analysis failed: {image_analysis.get('message', image_analysis.get('error'))}]"
            else:
                image_analysis_str = json.dumps(image_analysis, indent=2)
                enhanced_input += f"\n\nImage Analysis Context: {image_analysis_str}"

        # Generate response using convo chain
        response = self.conversational_chain.invoke(
            {
                "input": enhanced_input,
                "image_analysis": json.dumps(image_analysis) if image_analysis else ""
            },
            config={"configurable": {"session_id": user_id}}
        )

        # Track response
        self.fashion_db.track_user_behavior(user_id, "response_received", query=message)

        return {
            "answer": response["answer"],
            "recommendations": recommendations,
            "context_products": [doc.metadata for doc in response.get("context", [])],
            "user_preferences": self.rec_engine.analyze_user_preferences(user_id),
            "image_analysis": image_analysis
        }


# In[368]:


# ===== PART 9: INTERACTIVE CHAT INTERFACE =====
def enhanced_fashion_chat_interface():
    """Enhanced chat interface with image upload support"""

    print("\n" + "="*60)
    print("👗 Welcome to your Enhanced Personal Fashion Stylist! 👗")
    print("="*60)
    print("I can help with outfit recommendations, styling tips, and image analysis.")
    print("\nAvailable Commands:")
    print("💬 Type your message for fashion advice")
    print("📸 Type 'upload' to upload an image for analysis")
    print("📋 Type 'history' to see your uploaded images")
    print("❌ Type 'quit' to exit")
    print("="*60)

    # Get user ID
    user_id = input("\n🔑 Enter your user ID: ").strip()
    if not user_id:
        user_id = "default_user"

    print(f"\n👋 Hello {user_id}! I'm your fashion stylist assistant.")
    print("How can I help you look amazing today?")

    # Main chat loop
    while True:
        print("\n" + "-"*40)
        user_input = input(f"\n{user_id}: ").strip()

        if user_input.lower() == 'quit':
            print("\n✨ Thanks for chatting! Stay stylish! ✨")
            break

        elif user_input.lower() == 'upload':
            print("\n📸 IMAGE UPLOAD MODE")
            image_path = input("Enter the full path to your image file: ").strip()

            if not image_path:
                print("❌ No image path provided.")
                continue

            try:
                print("\n🔄 Processing your image...")

                # Handle image upload
                upload_result = enhanced_chatbot.handle_image_upload(
                    user_id, image_path, "Analyze this fashion image"
                )

                if upload_result["success"]:
                    print(f"\n✅ Image uploaded and analyzed successfully!")
                    print(f"📁 Saved as: {upload_result['image_path']}")

                    # Display analysis results
                    analysis = upload_result["analysis"]
                    print(f"\n🔍 ANALYSIS RESULTS:")
                    print("-" * 30)

                    if "error" not in analysis:
                        if "clothing_items" in analysis:
                            print(f"👕 Items detected: {analysis['clothing_items']}")
                        if "colors" in analysis:
                            print(f"🎨 Color palette: {analysis['colors']}")
                        if "style_analysis" in analysis:
                            print(f"✨ Style analysis: {analysis['style_analysis']}")
                        if "occasion" in analysis:
                            print(f"🎯 Suitable for: {analysis['occasion']}")
                    else:
                        print(f"⚠️ Analysis note: {analysis.get('message', 'Basic analysis completed')}")

                    # Get fashion advice based on image
                    print("\n💡 Now you can ask me questions about this outfit!")
                    follow_up = input("Ask me anything about this image: ").strip()

                    if follow_up:
                        print("\n🤖 Analyzing and responding...")
                        response = enhanced_chatbot.chat_with_image_context(
                            user_id, follow_up, upload_result["analysis"]
                        )
                        print(f"\n🎨 Fashion Stylist: {response['answer']}")

                        # Show recommendations if available
                        if response['recommendations']:
                            print("\n💡 PERSONALIZED RECOMMENDATIONS:")
                            print("-" * 35)
                            for i, rec in enumerate(response['recommendations'], 1):
                                print(f"{i}. {rec['name']} - ${rec['price']} (ID: {rec['product_id']})")

                else:
                    print(f"❌ Upload failed: {upload_result['error']}")
                    print("Please check your image path and try again.")

            except FileNotFoundError:
                print("❌ Image file not found. Please check the file path.")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif user_input.lower() == 'history':
            print("\n📋 YOUR IMAGE HISTORY:")
            images = enhanced_chatbot.fashion_db.get_user_images(user_id)
            if images:
                for i, img in enumerate(images, 1):
                    print(f"{i}. {img[3]} (Uploaded: {img[6]})")
            else:
                print("No images uploaded yet.")

        else:
            try:
                print("\n🤖 Thinking about your fashion question...")

                # Regular chat without image
                response = enhanced_chatbot.chat_with_image_context(user_id, user_input)

                print(f"\n🎨 Fashion Stylist: {response['answer']}")

                # Show recommendations
                if response['recommendations']:
                    print("\n💡 PERSONALIZED RECOMMENDATIONS:")
                    print("-" * 35)
                    for i, rec in enumerate(response['recommendations'], 1):
                        print(f"{i}. {rec['name']} - ${rec['price']} ({rec['brand']})")

                # Show user style profile
                if response['user_preferences']['preferred_colors']:
                    colors = ', '.join(response['user_preferences']['preferred_colors'])
                    print(f"\n🎨 Your Style Profile - Favorite Colors: {colors}")

            except Exception as e:
                print(f"❌ Sorry, I encountered an error: {e}")
                print("Please try rephrasing your question.")


# In[370]:


# ===== PART 10: MAIN EXECUTION (FIXED) =====

# Initialize all components
print("🚀 Initializing Enhanced Fashion Chatbot...")

# 1. Initialize database
print("\n1️⃣ Setting up database...")
fashion_db = FashionDatabase()

# 2. Initialize recommendation engine
print("2️⃣ Setting up recommendation engine...")
rec_engine = FashionRecommendationEngine(fashion_db)

# 3. Create vector store
print("3️⃣ Creating vector store...")
vectorstore = create_fashion_vector_store(fashion_db, openai_embed_model)

# 4. Create retriever
print("4️⃣ Setting up retriever...")
fashion_retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)

# 5. Initialize enhanced chatbot with FIXED parameters
print("5️⃣ Initializing enhanced chatbot...")
enhanced_chatbot = EnhancedFashionChatbot(
    chatgpt, fashion_retriever, fashion_db, rec_engine, openai_native_client  # FIXED: Added native client
)

print("\n✅ ALL SYSTEMS READY!")
print("🎉 Enhanced Fashion Chatbot with Image Upload is ready to use!")

# Start the chat interface
enhanced_fashion_chat_interface()


#
