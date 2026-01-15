"""
RAG Engine Module

This module handles the core logic for the Retrieval-Augmented Generation (RAG) system.
It utilizes Pinecone for vector storage and Google's Gemini mechanism for generation.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import PromptTemplate

# Load environment variables explicitly to ensure API keys are available
load_dotenv(override=True)

# Configuration Constants
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-knowledge-rag")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-flash-latest"

def get_rag_chain():
    """
    Initializes and reconstructs the RAG pipeline.

    This function performs the following steps:
    1. Validates the existence of necessary API keys.
    2. Initializes the HuggingFace embedding model.
    3. Connects to the existing Pinecone vector index.
    4. Sets up the Google Gemini LLM with specific parameters.
    5. Constructs the ConversationalRetrievalChain with a strict domain-specific prompt.

    Returns:
        tuple: A tuple containing (qa_chain, vectorstore).
    """
    
    # 1. API Key Validation
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not google_api_key:
        raise ValueError("Error: GOOGLE_API_KEY is missing in environment variables.")
    if not pinecone_api_key:
        raise ValueError("Error: PINECONE_API_KEY is missing in environment variables.")

    # 2. Initialize Embeddings
    # We use a lightweight, high-performance model suitable for semantic search
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 3. Connect to Pinecone Vector Store
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    
    # 4. Setup LLM (Gemini)
    # Temperature is set to 0 to maximize determinism and factual accuracy
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=google_api_key,
        temperature=0
    )
    
    # 5. Define Strict Prompt Template
    # This prompt enforces the agent to act as a specialized expert and refuse out-of-domain queries.
    template = """
    You are a specialized Artificial Intelligence Expert. 
    Your goal is to provide accurate information STRICTLY related to the field of Artificial Intelligence.
    
    RULES:
    1. Answer ONLY using the information provided in the given context.
    2. If the user’s question is NOT related to Artificial Intelligence OR the answer is NOT explicitly present in the context, respond exactly with:
    "I'm sorry, as an AI expert, I can only provide information strictly related to Artificial Intelligence. This topic is outside my current knowledge base."
    3. Do NOT invent, assume, infer, or add any information beyond the provided context.
    4. Use internal reasoning to arrive at the answer, but DO NOT reveal your chain of thought or intermediate reasoning steps.
    5. Provide only the final, concise answer.

    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    qa_prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )
    
    # 6. Create the Conversational Chain
    # We retrieve the top 3 most relevant chunks to serve as context
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )
    
    return qa_chain, vectorstore
