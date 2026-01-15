"""
RAG Chatbot API

This module provides a FastAPI-based REST API for the AI RAG Chatbot.
It exposes endpoints for chatting with the AI and checking service health.

Usage:
    Run the server using: python api.py
    Access documentation at: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_engine import get_rag_chain

# Initialize FastAPI app with metadata
app = FastAPI(
    title="AI Knowledge Base API",
    description="A specialized RAG API for Artificial Intelligence queries.",
    version="1.0.0"
)

# --- Pydantic Data Models ---

class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.
    Attributes:
        query (str): The user's question.
        chat_history (List[tuple]): Optional conversation history for context.
    """
    query: str
    chat_history: Optional[List[tuple]] = []

class SourceDocument(BaseModel):
    """
    Model representing a retrieved source document.
    """
    page_content: str
    metadata: dict

class ChatResponse(BaseModel):
    """
    Response model containing the answer and source documents.
    """
    answer: str
    sources: List[SourceDocument]

# --- Global State ---
rag_chain = None
vectorstore = None

# --- Event Handlers ---

@app.on_event("startup")
async def startup_event():
    """
    Initialize the RAG engine on application startup.
    This ensures we connect to Pinecone and load models only once.
    """
    global rag_chain, vectorstore
    try:
        print("Initializing RAG system...")
        rag_chain, vectorstore = get_rag_chain()
        print("RAG System initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize RAG system: {e}")

# --- Endpoints ---

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "AI RAG Chatbot API is running. Visit /docs for interactive documentation."}

@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    return {"status": "healthy", "service": "rag-chatbot"}

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Process a user query and return an AI-generated response.
    
    Args:
        request (ChatRequest): The user's query and optional history.
        
    Returns:
        ChatResponse: The AI answer and cited sources.
        
    Raises:
        HTTPException: If the RAG system is not initialized or an internal error occurs.
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")
    
    try:
        # Generate answer using the RAG chain
        response = rag_chain.invoke({
            "question": request.query,
            "chat_history": request.chat_history
        })
        
        # Extract and format the source documents for transparency
        sources = []
        if 'source_documents' in response:
            for doc in response['source_documents']:
                sources.append(SourceDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                ))
        
        return ChatResponse(
            answer=response['answer'],
            sources=sources
        )
        
    except Exception as e:
        # Log the error (in a real app, use a logger) and return 500
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server on localhost port 8000
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
