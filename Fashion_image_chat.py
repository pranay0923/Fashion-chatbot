# fashion_chatbot.py
# Enhanced Fashion Chatbot with Image Analysis - Complete Implementation

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
import openai
import re

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===== PART 2: DATABASE CLASS =====
class FashionDatabase:
    """Enhanced database class with image support"""
    
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fashion products table
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
        
        # User uploaded images table
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
        
        # User behavior tracking
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
        
        conn.commit()
        conn.close()
        
        # Load sample data if products table is empty
        self.load_sample_data_if_empty()
    
    def load_sample_data_if_empty(self):
        """Load sample data if products table is empty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            self.load_sample_products()
    
    def load_sample_products(self):
        """Load sample fashion products"""
        sample_products = [
            {
                'name': 'Classic Blue Jeans',
                'category': 'Bottoms',
                'subcategory': 'Jeans',
                'brand': "Levi's",
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
                'brand': 'H&M',
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
        
        for product in sample_products:
            self.add_product(**product)
        
        print(f"✅ Loaded {len(sample_products)} sample products into database!")
    
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

# ===== PART 3: IMAGE ANALYSIS SERVICE =====
class ImageAnalysisService:
    """Service for analyzing fashion images using AI"""
    
    def __init__(self, openai_native_client):
        self.openai_client = openai_native_client
    
    def analyze_fashion_image(self, image_bytes, user_query=None):
        """Analyze fashion items in uploaded image using GPT-4 Vision"""
        try:
            # Convert bytes to base64
            if isinstance(image_bytes, bytes):
                image_content = base64.b64encode(image_bytes).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{image_content}"
            else:
                return {"error": "Invalid image data format"}
            
            # Create detailed analysis prompt
            analysis_prompt = f"""
            Analyze this fashion image and provide detailed information in JSON format:
            
            {{
                "clothing_items": "List all visible clothing items",
                "colors": "Dominant colors and palette",
                "style_analysis": "Fashion style and aesthetic",
                "occasion": "Suitable occasions for this outfit",
                "season": "Appropriate season",
                "styling_tips": "Specific styling advice",
                "similar_items": "What to look for when shopping"
            }}
            
            User question: {user_query if user_query else "General fashion analysis"}
            """
            
            # Call OpenAI GPT-4 Vision API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
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
            
            # Try to parse as JSON, fallback to structured text
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_json = {
                    "raw_analysis": analysis_text,
                    "clothing_items": "See raw analysis",
                    "colors": "See raw analysis",
                    "style_analysis": "See raw analysis"
                }
            
            return analysis_json
        
        except Exception as e:
            return {
                "error": str(e), 
                "message": "Failed to analyze image. Please check your image and API key."
            }

# ===== PART 4: RECOMMENDATION ENGINE =====

# Updated get_personalized_recommendations method with better diversity, deduplication, and personalization

class FashionRecommendationEngine:
    """Improved Fashion recommendation engine"""

    def __init__(self, fashion_db, retriever=None):
        self.fashion_db = fashion_db
        self.retriever = retriever

    def get_personalized_recommendations(self, user_id, query, limit=5, exclude_ids=None):
        """Return improved recommendations with diversity and personalization"""
        if exclude_ids is None:
            exclude_ids = []

        try:
            all_products = self.fashion_db.get_all_products()
            if not all_products:
                return []

            query_lower = query.lower()
            filtered_products = []

            category_keywords = {
                'tops': ['shirt', 'top', 'blouse', 't-shirt', 'tshirt', 'tee'],
                'bottoms': ['jeans', 'pants', 'trousers', 'shorts', 'skirt'],
                'outerwear': ['jacket', 'coat', 'blazer', 'cardigan', 'hoodie'],
                'dresses': ['dress', 'gown', 'frock'],
                'footwear': ['shoes', 'sneakers', 'boots', 'sandals', 'heels'],
                'accessories': ['bag', 'purse', 'belt', 'scarf', 'jewelry']
            }

            occasion_keywords = {
                'work': ['work', 'office', 'professional', 'business', 'interview', 'formal'],
                'casual': ['casual', 'everyday', 'relaxed', 'weekend'],
                'party': ['party', 'night out', 'clubbing', 'evening'],
                'wedding': ['wedding', 'formal event', 'ceremony'],
                'sport': ['gym', 'workout', 'exercise', 'running', 'athletic', 'sport']
            }

            color_keywords = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'brown', 'gray', 'purple']

            # Track user interaction preferences (simplified)
            preferences = self.analyze_user_preferences(user_id)

            product_scores = []
            for product in all_products:
                if product[0] in exclude_ids:
                    continue

                score = 0
                product_text = f"{product[1]} {product[2]} {product[3]} {product[4]} {product[6]} {product[8]} {product[9]} {product[12]}".lower()

                # Keyword scoring
                for category, keywords in category_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        if any(kw in product_text for kw in keywords) or category in product_text:
                            score += 10

                for occasion, keywords in occasion_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        if occasion in product_text or any(kw in product_text for kw in keywords):
                            score += 8

                for color in color_keywords:
                    if color in query_lower and color in product_text:
                        score += 5
                    if color in preferences['preferred_colors'] and color in product_text:
                        score += 2

                if product[4] and product[4].lower() in query_lower:
                    score += 7

                for word in query_lower.split():
                    if len(word) > 2 and word in product_text:
                        score += 3

                for season in ['summer', 'winter', 'spring', 'fall', 'autumn']:
                    if season in query_lower and season in product_text:
                        score += 4

                if 'men' in query_lower and product[11] in ['men', 'unisex']:
                    score += 3
                elif 'women' in query_lower and product[11] in ['women', 'unisex']:
                    score += 3

                # Bonus for preferred styles
                for style in preferences['preferred_styles']:
                    if style in product_text:
                        score += 2

                product_scores.append((product, score))

            # Sort by score
            product_scores.sort(key=lambda x: x[1], reverse=True)

            # Deduplication and diversity enforcement
            seen_categories = set()
            seen_brands = set()
            recommendations = []

            for product, score in product_scores:
                if len(recommendations) >= limit:
                    break

                if product[2] in seen_categories:
                    continue
                if product[4] in seen_brands:
                    continue

                recommendations.append({
                    'product_id': product[0],
                    'name': product[1],
                    'category': product[2] if len(product) > 2 else 'Fashion Item',
                    'brand': product[4] if len(product) > 4 else 'Unknown Brand',
                    'price': product[5] if len(product) > 5 else 0,
                    'color': product[6] if len(product) > 6 else 'Various',
                    'description': product[8] if len(product) > 8 else 'Stylish fashion item',
                    'occasion': product[12] if len(product) > 12 else 'Various occasions'
                })

                seen_categories.add(product[2])
                seen_brands.add(product[4])

            return recommendations

        except Exception as e:
            print(f"Error in recommendation engine: {e}")
            return []

    def analyze_user_preferences(self, user_id):
        try:
            conn = sqlite3.connect(self.fashion_db.db_path)
            cursor = conn.cursor()
            cursor.execute('''SELECT query FROM user_behavior WHERE user_id = ? AND query IS NOT NULL ORDER BY timestamp DESC LIMIT 10''', (user_id,))
            queries = [row[0] for row in cursor.fetchall()]
            conn.close()

            all_queries = ' '.join(queries).lower()

            colors = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'brown']
            styles = ['casual', 'formal', 'sporty', 'elegant', 'trendy', 'classic']

            return {
                'preferred_colors': [c for c in colors if c in all_queries],
                'preferred_styles': [s for s in styles if s in all_queries],
                'budget_range': 'medium'
            }

        except Exception as e:
            print(f"Error analyzing preferences: {e}")
            return {
                'preferred_colors': [],
                'preferred_styles': [],
                'budget_range': 'medium'
            }


# ===== PART 5: FASHION CHATBOT =====
class FashionChatbot:
    """Main Fashion Chatbot class for API usage"""
    
    def __init__(self, chatgpt, retriever, fashion_db, rec_engine):
        self.chatgpt = chatgpt
        self.retriever = retriever
        self.fashion_db = fashion_db
        self.rec_engine = rec_engine
        
        # Initialize OpenAI native client for image analysis
        self.openai_native_client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.image_analyzer = ImageAnalysisService(self.openai_native_client)
        
        self.setup_chains()
        
        # Create uploads directory
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def setup_chains(self):
        """Setup LangChain conversation chains"""
        
        # Rephrase prompt
        rephrase_system_prompt = """
        You are a fashion stylist assistant. Given a conversation history and the latest 
        user question, formulate a standalone question that captures the user's fashion needs.
        """
        
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", rephrase_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        self.history_aware_retriever = create_history_aware_retriever(
            self.chatgpt, self.retriever, rephrase_prompt
        )
        
        # QA prompt
        qa_system_prompt = """
        You are an expert fashion stylist and personal shopping assistant.
        
        Provide helpful fashion advice based on the context and user questions.
        Be friendly, knowledgeable, and provide specific recommendations.
        
        Context: {context}
        """
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.chatgpt, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)
        
        # Memory management
        def get_session_history(session_id):
            return SQLChatMessageHistory(session_id, "sqlite:///fashion_memory.db")
        
        self.conversational_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def chat(self, user_id, message, image_bytes=None):
        """Main chat method for API usage"""
        try:
            # Track user query
            self.fashion_db.track_user_behavior(user_id, "query", query=message)
            
            # Handle image analysis if image provided
            image_analysis = None
            if image_bytes:
                print("🔍 Analyzing uploaded image...")
                image_analysis = self.image_analyzer.analyze_fashion_image(image_bytes, message)
                
                # Save image analysis to database
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"{self.upload_dir}/{user_id}_{timestamp}.jpg"
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                image_id = self.fashion_db.save_user_image(
                    user_id, image_path, message, 
                    image_analysis.get('clothing_items'),
                    None, image_analysis.get('style_analysis')
                )
                
                self.fashion_db.track_user_behavior(
                    user_id, "image_upload", image_id=image_id, query=message
                )
            
            # Check if user wants recommendations
            rec_keywords = ['recommend', 'suggest', 'show me', 'options', 'what should I buy']
            wants_recommendations = any(keyword in message.lower() for keyword in rec_keywords)
            
            recommendations = []
            if wants_recommendations:
                recommendations = self.rec_engine.get_personalized_recommendations(
                    user_id, message, limit=3
                )
            
            # Prepare enhanced input
            enhanced_input = message
            if image_analysis:
                enhanced_input += f"\n\nImage Analysis: {json.dumps(image_analysis, indent=2)}"
            
            # Generate response
            response = self.conversational_chain.invoke(
                {"input": enhanced_input},
                config={"configurable": {"session_id": user_id}}
            )
            
            # Track response
            self.fashion_db.track_user_behavior(user_id, "response_received", query=message)
            
            return {
                "success": True,
                "answer": response["answer"],
                "recommendations": recommendations,
                "image_analysis": image_analysis,
                "context_products": [doc.metadata for doc in response.get("context", [])]
            }
        
        except Exception as e:
            print(f"Error in chat method: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "Sorry, I encountered an error. Please try again."
            }

# ===== INTERACTIVE MODE (for direct running) =====
def interactive_mode():
    """Interactive mode when running the file directly"""
    print("🚀 Starting Fashion Chatbot in Interactive Mode...")
    
    # Setup OpenAI
    OPENAI_KEY = ""
    os.environ['OPENAI_API_KEY'] = OPENAI_KEY
    
    # Initialize components
    chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
    openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    fashion_db = FashionDatabase()
    
    # Create vector store
    print("📚 Creating vector store...")
    products = fashion_db.get_all_products()
    documents = []
    
    for product in products:
        doc_text = f"""
        Name: {product[1]}
        Category: {product[2]} - {product[3]}
        Brand: {product[4]}
        Price: ${product[5]}
        Color: {product[6]}
        Description: {product[8]}
        """
        
        doc = Document(
            page_content=doc_text.strip(),
            metadata={
                'product_id': product[0],
                'name': product[1],
                'category': product[2],
                'brand': product[4],
                'price': product[5]
            }
        )
        documents.append(doc)
    
    vectorstore = FAISS.from_documents(documents=documents, embedding=openai_embed_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Initialize chatbot
    rec_engine = FashionRecommendationEngine(fashion_db, retriever)
    chatbot = FashionChatbot(chatgpt, retriever, fashion_db, rec_engine)
    
    print("✅ Fashion Chatbot ready!")
    print("Type 'quit' to exit")
    
    user_id = input("Enter your user ID: ").strip() or "default_user"
    
    while True:
        user_input = input(f"\n{user_id}: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        # For interactive mode, we don't handle image uploads
        response = chatbot.chat(user_id, user_input)
        
        print(f"\n🎨 Fashion Stylist: {response['answer']}")
        
        if response.get('recommendations'):
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(response['recommendations'], 1):
                print(f"{i}. {rec['name']} - ${rec['price']} ({rec['brand']})")

if __name__ == "__main__":
    interactive_mode()
