# Specialized AI RAG Chatbot

An Artificial Intelligence assistant grounded in a strictly curated knowledge base.
This project uses Retrieval-Augmented Generation (RAG) to provide accurate, domain-specific answers about AI, Machine Learning, and Deep Learning, avoiding hallucinations by strictly adhering to retrieved context.

## 🚀 Key Features

*   **Strict Domain Control**: Answers are rigorously filtered to ensure they relate *only* to Artificial Intelligence.
*   **Hybrid RAG Architecture**: Combines vector search (semantic understanding) with strict prompt engineering.
*   **Professional Interfaces**:
    *   **Web Dashboard**: A modern Streamlit chat interface with source transparency.
    *   **REST API**: A robust FastAPI backend for programmatic access.
*   **Conversational Memory**: Maintains context for follow-up questions.

## 🛠️ Technology Stack

*   **LLM**: Google Gemini Flash
*   **Vector Database**: Pinecone (Serverless)
*   **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
*   **Frameworks**: LangChain, FastAPI, Streamlit

## 📂 Project Structure

```
├── app.py              # Streamlit Web Application (Frontend)
├── api.py              # FastAPI Server (Backend)
├── rag_engine.py       # Core RAG Logic & Chain Construction
├── setup_pinecone.py   # Database Initialization & Ingestion Script
├── ingest_data.py      # Wikipedia Data Fetching Module
├── .env                # Configuration (API Keys)
└── requirements.txt    # Project Dependencies
```

## ⚡ Quick Start

### 1. Setup Environment
Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=ai-knowledge-rag
```

### 3. Run the Web Interface
Launch the interactive chat dashboard:

```bash
streamlit run app.py
```

### 4. Run the API Server
Start the backend service:

```bash
python api.py
```
*Chat endpoint available at: http://127.0.0.1:8000/chat*

## 🧠 Knowledge Base Setup
(Only required once)
To re-ingest data from Wikipedia into the vector database:
```bash
python setup_pinecone.py
```
