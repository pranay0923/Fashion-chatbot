import os
import json
import sqlite3
import base64
from datetime import datetime
from PIL import Image

import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import re

# ===================== DATABASE =====================
class FashionDatabase:
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
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

    def add_product(self, **product_data):
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products")
        products = cursor.fetchall()
        conn.close()
        return products

    def save_user_image(self, user_id, image_path, description=None, detected_items=None, 
                       color_analysis=None, style_analysis=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO user_images (user_id, image_path, image_description, detected_items, color_analysis, style_analysis)
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

# =============== IMAGE ANALYSIS SERVICE ==============
class ImageAnalysisService:
    def __init__(self, openai_native_client):
        self.openai_client = openai_native_client

    def analyze_fashion_image(self, image_path_or_base64, user_query=None):
        try:
            # Prepare the image data
            if image_path_or_base64.startswith('data:image'):
                image_data = image_path_or_base64
            else:
                with open(image_path_or_base64, "rb") as image_file:
                    image_content = base64.b64encode(image_file.read()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{image_content}"
            analysis_prompt = f"""
            Analyze this fashion image and provide detailed information in the following categories:
            1. Clothing Items; 2. Colors; 3. Style Analysis; 4. Occasion; 5. Season; 6. Body Type; 7. Styling Tips; 8. Similar Items.
            User's specific question: {user_query if user_query else "General fashion analysis"}
            Respond in JSON with these as keys.
            """
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]}],
                max_tokens=1000
            )
            analysis_text = response.choices[0].message.content
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_json = {
                    "raw_analysis": analysis_text,
                    "clothing_items": "See 'raw_analysis'",
                    "colors": "See 'raw_analysis'",
                    "style_analysis": "See 'raw_analysis'"
                }
            return analysis_json
        except Exception as e:
            return {"error": str(e), "message": "Failed to analyze image."}

    def extract_colors_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            colors = image.getcolors(maxcolors=256*256*256)
            if colors:
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

# =========== RECOMMENDATION ENGINE ===========
class FashionRecommendationEngine:
    def __init__(self, fashion_db):
        self.fashion_db = fashion_db
    def get_personalized_recommendations(self, user_id, query, limit=5):
        try:
            products = self.fashion_db.get_all_products()
            if not products:
                return []
            recommendations = []
            for product in products[:limit]:
                rec = {'product_id': product[0],'name': product[1],'category': product[2],'brand': product[4],'price': product[5],'color': product[6],'description': product[8]}
                recommendations.append(rec)
            return recommendations
        except Exception as e:
            return []
    def analyze_user_preferences(self, user_id):
        return {
            'preferred_colors': ['blue', 'black', 'white'],
            'preferred_brands': ['Nike', 'Adidas'],
            'preferred_styles': ['casual', 'sporty'],
            'budget_range': 'medium'
        }

# ============== VECTOR STORE SETUP =================
def create_fashion_vector_store(fashion_db, embeddings_model):
    products = fashion_db.get_all_products()
    if not products:
        load_sample_fashion_data(fashion_db)
        products = fashion_db.get_all_products()
    documents = []
    for product in products:
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
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings_model
    )
    vectorstore.save_local("fashion_faiss_db")
    return vectorstore

# =============== LOAD SAMPLE DATA ============
def load_sample_fashion_data(fashion_db):
    sample_products = [
        # ... [list your same sample products here, from your original script] ...
    ]
    for product in sample_products:
        fashion_db.add_product(**product)
    return sample_products

# =========== ENHANCED CHATBOT CLASS ===========
class EnhancedFashionChatbot:
    def __init__(self, chatgpt, retriever, fashion_db, rec_engine, openai_native_client):
        self.chatgpt = chatgpt
        self.retriever = retriever
        self.fashion_db = fashion_db
        self.rec_engine = rec_engine
        self.image_analyzer = ImageAnalysisService(openai_native_client)
        self.setup_chains()
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    def setup_chains(self):
        rephrase_system_prompt = """
        You are a fashion stylist assistant with image analysis capabilities.
        Given a conversation history and the latest user question (which might include image analysis results),
        formulate a standalone question that captures the user's fashion needs.
        """
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", rephrase_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.history_aware_retriever = create_history_aware_retriever(
            self.chatgpt, self.retriever, rephrase_prompt
        )
        qa_system_prompt = """
        You are an expert fashion stylist and personal shopping assistant with image analysis capabilities.
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
        def get_session_history(session_id):
            return SQLChatMessageHistory(session_id, "sqlite:///fashion_memory.db")
        self.conversational_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    def handle_image_upload(self, user_id, image_file_path, query=None):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}_{timestamp}.jpg"
            image_path = os.path.join(self.upload_dir, filename)
            with open(image_file_path, 'rb') as src, open(image_path, 'wb') as dst:
                dst.write(src.read())
            image_analysis = self.image_analyzer.analyze_fashion_image(image_path, query)
            color_analysis = self.image_analyzer.extract_colors_from_image(image_path)
            image_id = self.fashion_db.save_user_image(
                user_id=user_id,
                image_path=image_path,
                description=query,
                detected_items=image_analysis.get('clothing_items') if isinstance(image_analysis, dict) else None,
                color_analysis=color_analysis,
                style_analysis=image_analysis.get('style_analysis') if isinstance(image_analysis, dict) else None
            )
            self.fashion_db.track_user_behavior(
                user_id, "image_upload", image_id=image_id, query=query
            )
            return {
                "success": True,
                "image_id": image_id,
                "image_path": image_path,
                "analysis": image_analysis,
                "colors": color_analysis
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    def chat_with_image_context(self, user_id, message, image_analysis=None):
        self.fashion_db.track_user_behavior(user_id, "query", query=message)
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
        if rec_query:
            recommendations = self.rec_engine.get_personalized_recommendations(
                user_id, message, limit=3
            )
        else:
            recommendations = []
        enhanced_input = message
        if image_analysis:
            if isinstance(image_analysis, dict) and image_analysis.get("error"):
                enhanced_input += f"\n\n[Image analysis failed: {image_analysis.get('message', image_analysis.get('error'))}]"
            else:
                image_analysis_str = json.dumps(image_analysis, indent=2)
                enhanced_input += f"\n\nImage Analysis Context: {image_analysis_str}"
        response = self.conversational_chain.invoke(
            {
                "input": enhanced_input,
                "image_analysis": json.dumps(image_analysis) if image_analysis else ""
            },
            config={"configurable": {"session_id": user_id}}
        )
        self.fashion_db.track_user_behavior(user_id, "response_received", query=message)
        return {
            "answer": response["answer"],
            "recommendations": recommendations,
            "context_products": [doc.metadata for doc in response.get("context", [])],
            "user_preferences": self.rec_engine.analyze_user_preferences(user_id),
            "image_analysis": image_analysis
        }

# ========== FACTORY FUNCTION TO CREATE CHATBOT =========
def get_fashion_chatbot(openai_api_key: str = None):
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY must be set.")
    fashion_db = FashionDatabase()
    rec_engine = FashionRecommendationEngine(fashion_db)
    chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = create_fashion_vector_store(fashion_db, embed_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    openai_native_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chatbot = EnhancedFashionChatbot(chatgpt, retriever, fashion_db, rec_engine, openai_native_client)
    return chatbot
