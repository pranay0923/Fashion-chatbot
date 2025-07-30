import os
import sqlite3
import json
from datetime import datetime
from PIL import Image
import io

# You must install langchain and other deps
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behavior (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            product_id INTEGER,
            query TEXT,
            preferences TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (id)
        )''')
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
        )''')
        conn.commit()
        conn.close()

    def add_product(self, product_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO products 
            (name, category, subcategory, brand, price, color, size, description, style_tags, season, gender, occasion, material, image_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', product_data)
        conn.commit()
        conn.close()

    def track_user_behavior(self, user_id, action_type, product_id=None, query=None, preferences=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO user_behavior (user_id, action_type, product_id, query, preferences)
        VALUES (?, ?, ?, ?, ?)''', (user_id, action_type, product_id, query, json.dumps(preferences) if preferences else None))
        conn.commit()
        conn.close()

    def get_user_behavior_data(self, user_id, limit=20):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT ub.*, p.name, p.category, p.brand, p.color, p.style_tags
        FROM user_behavior ub
        LEFT JOIN products p ON ub.product_id = p.id
        WHERE ub.user_id = ?
        ORDER BY ub.timestamp DESC
        LIMIT ?''', (user_id, limit))
        result = cursor.fetchall()
        conn.close()
        return result

class FashionRecommendationEngine:
    def __init__(self, fashion_db, vector_retriever):
        self.fashion_db = fashion_db
        self.vector_retriever = vector_retriever

    def analyze_user_preferences(self, user_id):
        behavior_data = self.fashion_db.get_user_behavior_data(user_id, limit=50)
        preferences = {
            "preferred_colors": [],
            "preferred_brands": [],
            "preferred_categories": [],
            "preferred_styles": [],
            "price_range": {"min": 0, "max": 1000},
            "interaction_patterns": {}
        }
        for record in behavior_data:
            product_id = record[3]
            if product_id:
                color = record[8]
                brand = record[7]
                category = record[6]
                if color: preferences["preferred_colors"].append(color)
                if brand: preferences["preferred_brands"].append(brand)
                if category: preferences["preferred_categories"].append(category)
        for key in ["preferred_colors", "preferred_brands", "preferred_categories"]:
            preferences[key] = list(set(preferences[key]))[:3]
        return preferences

    def get_personalized_recommendations(self, user_id, query, limit=5):
        user_preferences = self.analyze_user_preferences(user_id)
        enhanced_query = f"{query} preferred colors: {', '.join(user_preferences['preferred_colors'])} preferred brands: {', '.join(user_preferences['preferred_brands'])}"
        similar_docs = self.vector_retriever.invoke(enhanced_query)
        recommendations = []
        for doc in similar_docs[:limit]:
            recommendations.append({
                "product_id": doc.metadata.get("product_id"),
                "name": doc.metadata.get("name"),
                "content": doc.page_content,
                "relevance_score": "high",
            })
        return recommendations

class FashionChatbot:
    def __init__(self, chatgpt, retriever, fashion_db, rec_engine):
        self.chatgpt = chatgpt
        self.retriever = retriever
        self.fashion_db = fashion_db
        self.rec_engine = rec_engine
        self.setup_chains()

    def setup_chains(self):
        rephrase_system_prompt = (
            "You are a fashion stylist assistant..."
        )
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", rephrase_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.history_aware_retriever = create_history_aware_retriever(self.chatgpt, self.retriever, rephrase_prompt)
        qa_system_prompt = (
            "You are an expert fashion stylist and personal shopping assistant..."
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.chatgpt, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)
        def get_session_history(session_id, topk_conversations=3):
            return SQLChatMessageHistory(session_id, "sqlite:///fashion_memory.db")
        self.conversational_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def analyze_image(self, image_bytes):
        # Simple: read with PIL, return info; advanced: plug in model inference here
        if not image_bytes:
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes))
            info = f"Uploaded image resolution: {img.size}, format: {img.format}"
            # Plug in ML vision model here for deep fashion info
            return info
        except Exception as e:
            return f"Could not process image: {e}"

    def chat(self, user_id, message, image_bytes=None):
        # Log user query
        self.fashion_db.track_user_behavior(user_id, "query", query=message)
        recommendations = self.rec_engine.get_personalized_recommendations(user_id, message, limit=3)
        # Image analysis (if any)
        image_info = self.analyze_image(image_bytes) if image_bytes else None
        # Optionally append image_info into the system/context prompt or answer
        augmented_message = message
        if image_info:
            augmented_message += f"\n[Image analysis: {image_info}]"
        # Generate response
        response = self.conversational_chain.invoke(
            {"input": augmented_message},
            config={"configurable": {"session_id": user_id}}
        )
        self.fashion_db.track_user_behavior(user_id, "response_received", query=message)
        answer = response["answer"]
        if image_info:
            answer += f"\n\n_Image info: {image_info}_"
        return {
            "answer": answer,
            "recommendations": recommendations,
            "user_preferences": self.rec_engine.analyze_user_preferences(user_id),
        }
