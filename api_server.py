# api_server.py

# --- Imports from your notebook ---
import os
import json
import sqlite3
from getpass import getpass
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.docstore.document import Document

# --- Setup OpenAI API Key ---
# For security, it's best to set this as an environment variable.
# You can uncomment the getpass line for testing if you prefer.
# os.environ['OPENAI_API_KEY'] = getpass("Enter your API Token here: ")
if 'OPENAI_API_KEY' not in os.environ:
    print("🚨 OPENAI_API_KEY environment variable not set. Please set it before running.")
    exit()

# --- Initialize Models (from notebook) ---
chatgpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

# --- All Classes from your Notebook (FashionDatabase, FashionRecommendationEngine, FashionChatbot) ---
# I've copied them here directly for completeness.

class FashionDatabase:
    def __init__(self, db_path="fashion_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT, subcategory TEXT,
            brand TEXT, price REAL, color TEXT, size TEXT, description TEXT,
            style_tags TEXT, season TEXT, gender TEXT, occasion TEXT, material TEXT,
            image_url TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behavior (
            id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, action_type TEXT NOT NULL,
            product_id INTEGER, query TEXT, preferences TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (id)
        )''')
        conn.commit()
        conn.close()

    def add_product(self, product_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO products (id, name, category, subcategory, brand, price, color, size, description, style_tags, season, gender, occasion, material, image_url) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', product_data)
        conn.commit()
        conn.close()

    def track_user_behavior(self, user_id, action_type, product_id=None, query=None, preferences=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO user_behavior (user_id, action_type, product_id, query, preferences) VALUES (?, ?, ?, ?, ?)',
                       (user_id, action_type, product_id, query, json.dumps(preferences) if preferences else None))
        conn.commit()
        conn.close()

    def get_user_behavior_data(self, user_id, limit=20):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ub.*, p.name, p.category, p.brand, p.color, p.style_tags
            FROM user_behavior ub LEFT JOIN products p ON ub.product_id = p.id
            WHERE ub.user_id = ? ORDER BY ub.timestamp DESC LIMIT ?
        ''', (user_id, limit))
        # Fetchall returns a list of tuples. We need to convert it to a list of dicts for easier access.
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

class FashionRecommendationEngine:
    def __init__(self, fashion_db, vector_retriever):
        self.fashion_db = fashion_db
        self.vector_retriever = vector_retriever
    
    def analyze_user_preferences(self, user_id):
        behavior_data = self.fashion_db.get_user_behavior_data(user_id, limit=50)
        preferences = {"preferred_colors": [], "preferred_brands": [], "preferred_categories": []}
        for record in behavior_data:
            if record.get("product_id"):
                if record.get("color"): preferences["preferred_colors"].append(record["color"])
                if record.get("brand"): preferences["preferred_brands"].append(record["brand"])
                if record.get("category"): preferences["preferred_categories"].append(record["category"])
        for key in preferences:
            preferences[key] = list(set(preferences[key]))[:3]
        return preferences

    def get_personalized_recommendations(self, user_id, query, limit=3):
        user_preferences = self.analyze_user_preferences(user_id)
        enhanced_query = f"{query} preferred colors: {', '.join(user_preferences['preferred_colors'])} preferred brands: {', '.join(user_preferences['preferred_brands'])}"
        similar_docs = self.vector_retriever.invoke(enhanced_query)
        recommendations = [{"product_id": doc.metadata.get("product_id"), "name": doc.metadata.get("name")} for doc in similar_docs[:limit]]
        return recommendations

class FashionChatbot:
    def __init__(self, chatgpt, retriever, fashion_db, rec_engine):
        self.chatgpt = chatgpt
        self.retriever = retriever
        self.fashion_db = fashion_db
        self.rec_engine = rec_engine
        self.setup_chains()
    
    def setup_chains(self):
        rephrase_system_prompt = "Given a conversation history and a user question, formulate a standalone fashion-related question."
        rephrase_prompt = ChatPromptTemplate.from_messages([("system", rephrase_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        self.history_aware_retriever = create_history_aware_retriever(self.chatgpt, self.retriever, rephrase_prompt)
        
        qa_system_prompt = "You are an expert fashion stylist. Use the retrieved fashion product info to give personalized advice. Context: {context}"
        qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(self.chatgpt, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)
        
        def get_session_history(session_id):
            return SQLChatMessageHistory(session_id, "sqlite:///fashion_memory.db")
        
        self.conversational_chain = RunnableWithMessageHistory(
            self.rag_chain, get_session_history,
            input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
        )

    def chat(self, user_id, message):
        self.fashion_db.track_user_behavior(user_id, "query", query=message)
        recommendations = self.rec_engine.get_personalized_recommendations(user_id, message, limit=3)
        response = self.conversational_chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": user_id}}
        )
        return {"answer": response["answer"], "recommendations": recommendations}

# --- Initialization Logic (from notebook) ---
fashion_db = FashionDatabase()

def load_fashion_data():
    sample_products = [
        ("Classic Denim Jacket", "Outerwear", "Jackets", "Levi's", 89.99, "Blue", "M", "A timeless denim jacket.", "casual,classic", "All Season", "Unisex", "Casual", "Cotton", ""),
        ("Floral Summer Dress", "Dresses", "Midi", "Zara", 59.99, "Pink", "S", "A light floral midi dress.", "feminine,floral", "Summer", "Women", "Casual", "Polyester", ""),
        ("Business Suit Set", "Suits", "Two-piece", "Hugo Boss", 299.99, "Navy", "L", "A professional two-piece suit.", "professional,formal", "All Season", "Men", "Business", "Wool", ""),
        ("Vintage Leather Boots", "Shoes", "Boots", "Dr. Martens", 149.99, "Black", "9", "Classic leather combat boots.", "edgy,vintage", "Fall", "Unisex", "Casual", "Leather", "")
    ]
    for product in sample_products:
        fashion_db.add_product(product)

def create_vector_db():
    conn = sqlite3.connect("fashion_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    fashion_docs = []
    for p in products:
        content = f"Product: {p[1]}, Category: {p[2]}, Brand: {p[4]}, Price: ${p[5]}, Description: {p[8]}, Style: {p[9]}"
        doc = Document(page_content=content, metadata={"product_id": p[0], "name": p[1], "brand": p[4]})
        fashion_docs.append(doc)
    conn.close()
    
    if not fashion_docs:
        return None
        
    vector_db = Chroma.from_documents(
        documents=fashion_docs, 
        embedding=openai_embed_model,
        persist_directory="./fashion_vector_db"
    )
    return vector_db.as_retriever(search_kwargs={"k": 5})

print("Loading initial data and creating vector database...")
load_fashion_data()
fashion_retriever = create_vector_db()

if fashion_retriever:
    rec_engine = FashionRecommendationEngine(fashion_db, fashion_retriever)
    fashion_chatbot = FashionChatbot(chatgpt, fashion_retriever, fashion_db, rec_engine)
    print("✅ Backend is ready.")
else:
    print("❌ Could not initialize retriever. No data found.")
    fashion_chatbot = None


# --- FastAPI Application ---
app = FastAPI(title="Fashion Chatbot API")

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not fashion_chatbot:
        return {"error": "Chatbot is not initialized."}
    response = fashion_chatbot.chat(req.user_id, req.message)
    return response

@app.get("/")
def root():
    return {"message": "Fashion Chatbot API is running. Use the /chat endpoint to interact."}

