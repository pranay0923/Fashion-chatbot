import os
import json
import sqlite3
import os
import openai
from datetime import datetime
from getpass import getpass

# Import langchain components, adjust imports to your environment
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Fashion_chatbot.py
import os
import openai

# ✅ Load the API key securely from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful fashion assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response["choices"][0]["message"]["content"]
        return reply
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


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
        CREATE TABLE IF NOT EXISTS user_behavior (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            product_id INTEGER,
            query TEXT,
            preferences TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (id)
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

    def add_product(self, product_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO products 
            (name, category, subcategory, brand, price, color, size, description, style_tags, season, gender, occasion, material, image_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', product_data)
        conn.commit()
        conn.close()

    def track_user_behavior(self, user_id, action_type, product_id=None, query=None, preferences=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO user_behavior (user_id, action_type, product_id, query, preferences)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, action_type, product_id, query, json.dumps(preferences) if preferences else None))
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
        LIMIT ?
        ''', (user_id, limit))
        result = cursor.fetchall()
        conn.close()
        return result


def load_fashion_data(fashion_db):
    sample_products = [
        ("Classic Denim Jacket", "Outerwear", "Jackets", "Levi's", 89.99, "Blue", "M",
         "A timeless denim jacket perfect for casual layering", "casual,classic,versatile",
         "All Season", "Unisex", "Casual", "Cotton", "https://example.com/denim-jacket.jpg"),

        ("Floral Summer Dress", "Dresses", "Midi", "Zara", 59.99, "Pink", "S",
         "Light and breezy midi dress with floral print, perfect for summer outings",
         "feminine,floral,romantic", "Summer", "Women", "Casual,Date", "Polyester",
         "https://example.com/floral-dress.jpg"),

        ("Business Suit Set", "Suits", "Two-piece", "Hugo Boss", 299.99, "Navy", "L",
         "Professional two-piece suit perfect for business meetings and formal events",
         "professional,formal,elegant", "All Season", "Men", "Business,Formal", "Wool",
         "https://example.com/business-suit.jpg"),

        ("Bohemian Maxi Dress", "Dresses", "Maxi", "Free People", 128.00, "Burgundy", "M",
         "Flowing maxi dress with bohemian prints and bell sleeves",
         "bohemian,flowy,artistic", "Summer", "Women", "Festival,Casual", "Rayon",
         "https://example.com/boho-dress.jpg"),

        ("Vintage Leather Boots", "Shoes", "Boots", "Dr. Martens", 149.99, "Black", "9",
         "Classic leather combat boots with vintage appeal",
         "edgy,vintage,durable", "Fall", "Unisex", "Casual,Alternative", "Leather",
         "https://example.com/leather-boots.jpg"),
    ]

    for product in sample_products:
        fashion_db.add_product(product)


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
                if color:
                    preferences["preferred_colors"].append(color)
                if brand:
                    preferences["preferred_brands"].append(brand)
                if category:
                    preferences["preferred_categories"].append(category)

        # Deduplicate and limit to top 3
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
                "relevance_score": "high",  # Placeholder for actual scoring logic
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
        rephrase_system_prompt = \
            """You are a fashion stylist assistant. Given a conversation history and the latest user question 
            which might reference previous context, formulate a standalone question that captures the user's 
            fashion needs, style preferences, and context from the conversation.
            Consider factors like: style preference, body type, occasion, budget, color preferences, 
            brand preferences, and seasonal needs."""

        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", rephrase_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.history_aware_retriever = create_history_aware_retriever(
            self.chatgpt, self.retriever, rephrase_prompt
        )

        qa_system_prompt = \
            """You are an expert fashion stylist and personal shopping assistant with deep knowledge of:
            - Fashion trends and seasonal styles
            - Body types and flattering fits
            - Color theory and coordination
            - Brand knowledge and quality assessment
            - Occasion-appropriate dressing
            - Budget-conscious styling

            Use the retrieved fashion product information to provide personalized styling advice.
            Always consider the user's preferences, budget, body type, and lifestyle.

            For each recommendation:
            1. Explain WHY it suits the user
            2. Suggest styling tips
            3. Mention care instructions if relevant
            4. Provide alternatives in different price ranges

            Keep responses conversational, encouraging, and style-focused.
            If you don't have specific product information, provide general styling advice.

            Context: {context}
            """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

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

    def chat(self, user_id, message):
        # Track user query
        self.fashion_db.track_user_behavior(
            user_id,
            "query",
            query=message
        )

        # Personalized recommendations
        recommendations = self.rec_engine.get_personalized_recommendations(user_id, message, limit=3)

        # Generate response
        response = self.conversational_chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": user_id}}
        )

        # Track the response
        self.fashion_db.track_user_behavior(
            user_id,
            "response_received",
            query=message
        )

        return {
            "answer": response["answer"],
            "recommendations": recommendations,
            "context_products": [doc.metadata for doc in response.get("context", [])],
            "user_preferences": self.rec_engine.analyze_user_preferences(user_id),
        }


def main():
    print("👗 Welcome to your Personal Fashion Stylist! 👗")
    print("I can help you with outfit recommendations, styling tips, and fashion advice.")
    print("Type 'quit' to exit.\n")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Securely enter your OpenAI API key
    OPENAI_KEY = getpass("Enter your OpenAI API Key: ")
    os.environ['OPENAI_API_KEY'] = OPENAI_KEY

    chatgpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Setup fashion database and load products
    fashion_db = FashionDatabase()
    load_fashion_data(fashion_db)

    # Create documents for vector database
    fashion_docs = []
    conn = sqlite3.connect(fashion_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    conn.close()

    for product in products:
        content = f"""
        Product: {product[1]}
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
            page_content=content,
            metadata={
                "product_id": product[0],
                "name": product[1],
                "category": product[2],
                "brand": product[4],
                "price": product[5],
            }
        )
        fashion_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = splitter.split_documents(fashion_docs)

    fashion_db_vector = Chroma.from_documents(
        documents=chunked_docs,
        collection_name='fashion_db',
        embedding=openai_embed_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./fashion_vector_db"
    )

    fashion_retriever = fashion_db_vector.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.3}
    )

    rec_engine = FashionRecommendationEngine(fashion_db, fashion_retriever)
    fashion_chatbot = FashionChatbot(chatgpt, fashion_retriever, fashion_db, rec_engine)

    user_id = input("Enter your user ID: ").strip()

    while True:
        user_input = input(f"\n{user_id}: ").strip()
        if user_input.lower() == "quit":
            print("Thanks for chatting! Stay stylish! ✨")
            break
        try:
            response = fashion_chatbot.chat(user_id, user_input)
            print(f"\n🤖 Fashion Stylist: {response['answer']}")
            if response['recommendations']:
                print("\n💡 Personalized Recommendations:")
                for i, rec in enumerate(response['recommendations'], 1):
                    print(f"{i}. {rec['name']} (ID: {rec['product_id']})")
            if response['user_preferences']['preferred_colors']:
                print(f"\n🎨 Your Style Profile: Colors: {', '.join(response['user_preferences']['preferred_colors'])}")
        except Exception as e:
            print(f"Sorry, I encountered an error: {e}")


if __name__ == "__main__":
    main()
