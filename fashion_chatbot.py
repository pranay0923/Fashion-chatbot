import os
import sqlite3
import json
from PIL import Image
import io

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
        # Table creation ... (same as previously provided)
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
        # ... other tables omitted for brevity ...
        conn.commit()
        conn.close()
    # ... (Other methods unchanged) ...

class FashionRecommendationEngine:
    def __init__(self, fashion_db, vector_retriever):
        self.fashion_db = fashion_db
        self.vector_retriever = vector_retriever
    # ... (Methods unchanged) ...

class FashionChatbot:
    def __init__(self, chatgpt, retriever, fashion_db, rec_engine):
        self.chatgpt = chatgpt
        self.retriever = retriever
        self.fashion_db = fashion_db
        self.rec_engine = rec_engine
        self.setup_chains()

    def setup_chains(self):
        rephrase_system_prompt = "You are a fashion stylist assistant..."
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", rephrase_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.history_aware_retriever = create_history_aware_retriever(
            self.chatgpt, self.retriever, rephrase_prompt
        )
        qa_system_prompt = (
            "You are an expert fashion stylist and personal shopping assistant..."
        )
        # ==== FIX: Add {context} as an input variable! ====
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            # Often {context} goes as a separate message for LangChain V2/V3
            ("system", "{context}"),
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
        if not image_bytes:
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return f"Uploaded image resolution: {img.size}, format: {img.format}"
        except Exception as e:
            return f"Could not process image: {e}"
    def chat(self, user_id, message, image_bytes=None):
        # ... logic as previously described ...
        image_info = self.analyze_image(image_bytes) if image_bytes else None
        augmented_message = message
        if image_info:
            augmented_message += f"\n[Image analysis: {image_info}]"
        response = self.conversational_chain.invoke(
            {"input": augmented_message},
            config={"configurable": {"session_id": user_id}}
        )
        answer = response["answer"]
        if image_info:
            answer += f"\n\n_Image info: {image_info}_"
        return {
            "answer": answer
        }
