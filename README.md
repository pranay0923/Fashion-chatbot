# 🧠 Fashion Chatbot Backend

This is the **backend API** for the Fashion Chatbot project, built using **FastAPI**. It processes user queries and returns intelligent fashion-related suggestions using LLMs and vector search.

---

## 🚀 Features

* FastAPI-based RESTful API
* Handles chat interactions via POST requests
* Uses `langchain`, `ChromaDB`, and OpenAI's GPT
* Modular design for future improvements (e.g., vector DB, memory, etc.)
* Ready to deploy on Render

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **Framework**: FastAPI
* **Vector Store**: ChromaDB
* **LLM Interface**: LangChain + OpenAI API
* **Deployment**: Render / Localhost

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚨 Environment Variables

Before running, set the following environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

You can use `.env` file support with packages like `python-dotenv` if preferred.

---

## ▶️ How to Run Locally

```bash
uvicorn api_server:app --reload
```

By default, the app runs at: `http://127.0.0.1:8000`

---

## 🧪 API Usage

### POST `/chat`

**Request Body:**

```json
{
  "user_id": "example_user",
  "message": "What should I wear to a beach party?"
}
```

**Response:**

```json
{
  "answer": "You can go for a floral shirt with white shorts and sandals."
}
```

---

## 🗂️ File Structure

```
fashion-chatbot-backend/
├── api_server.py         # FastAPI app
├── chatbot_engine.py     # Handles LLM + vector DB interaction
├── requirements.txt      # Dependencies
├── .env (optional)       # Environment variable storage
├── start.sh              # Script for deployment
```

---

## 🚀 Deployment on Render

1. Push your backend code to GitHub
2. Go to [https://render.com](https://render.com)
3. Create a new Web Service
4. Use `api_server:app` as the entry point
5. Add environment variable: `OPENAI_API_KEY`
6. Choose Python 3.10+ and deploy

---

## 🤝 Contact

For backend-related queries, connect via [LinkedIn](https://www.linkedin.com/in/mudigondapranay)

---

## 📄 License

MIT License
