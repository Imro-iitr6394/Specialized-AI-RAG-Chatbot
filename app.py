"""
RAG Chatbot Web Interface

This module runs a Streamlit-based web application for interacting with the AI.
It provides a user-friendly chat interface with transparency features.

Usage:
    Run the app using: streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import get_rag_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="Specialized AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
# Inject custom CSS for a cleaner, more modern look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stSpinner {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Initialization ---

# Load environment variables
load_dotenv(override=True)

# Validate API Keys early
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY is missing. Please check your .env file.")
    st.stop()

@st.cache_resource
def cached_rag_chain():
    """
    Cache the RAG chain initialization to prevent reloading models on every interaction.
    """
    return get_rag_chain()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("System Settings")
    
    index_name = os.getenv("PINECONE_INDEX_NAME", "ai-knowledge-rag")
    st.caption(f"Knowledge Base: **{index_name}**")
    st.caption("Model: **Gemini Flash**")
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.info(
        "**About:**\n"
        "This assistant answers questions strictly based on its internal knowledge base "
        "of Artificial Intelligence concepts."
    )

# --- Main Chat UI ---

st.title("🤖 Specialized AI Assistant")
st.markdown("ask me anything about *Artificial Intelligence, Machine Learning, or Deep Learning*.")
st.divider()

# 1. Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this message has source documents attached, show them in an expander
        if message.get("sources"):
            with st.expander("🔍 View Retrieved Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** *{doc.metadata.get('title', 'Unknown')}*")
                    st.caption(f"{doc.page_content[:300]}...")

# 2. Handle User Input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to UI immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        try:
            qa_chain, _ = cached_rag_chain()
            
            with st.spinner("Analyzing knowledge base..."):
                # Call the RAG engine
                response = qa_chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                })
            
            answer = response['answer']
            sources = response.get('source_documents', [])
            
            # Display Answer
            st.markdown(answer)
            
            # Display Sources
            if sources:
                with st.expander("🔍 View Retrieved Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** *{doc.metadata.get('title', 'Unknown')}*")
                        st.caption(f"{doc.page_content[:300]}...")
            
            # Update Session State
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            st.session_state.chat_history.append((prompt, answer))
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

