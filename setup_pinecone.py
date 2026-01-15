"""
Pinecone Database Setup Script

This script initializes the Pinecone vector index and ingests data into it.
It should be run once to populate the knowledge base.
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from ingest_data import fetch_and_chunk_ai_data

# Load environment variables
load_dotenv(override=True)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-knowledge-rag")
DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

def setup_pinecone():
    """
    Connects to Pinecone and creates the index if it doesn't exist.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, if not create it
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print("Waiting for index to serve...")
            time.sleep(1)
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
            
    return pc.Index(INDEX_NAME)

def upload_to_pinecone(index, chunks):
    """
    Encodes text chunks and uploads them to the Pinecone index.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Uploading {len(chunks)} chunks to Pinecone...")
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Prepare vectors
        # ID format: title_chunk_id
        ids = [f"{c['metadata']['title']}_{c['metadata']['chunk_id']}" for c in batch]
        texts = [c['text'] for c in batch]
        
        # Generate Embeddings
        embeddings = model.encode(texts).tolist()
        
        # Prepare Metadata
        metadata = []
        for c in batch:
            m = c['metadata']
            m['text'] = c['text'] # Store text in metadata for retrieval
            metadata.append(m)
            
        # Upsert to Pinecone
        index.upsert(vectors=zip(ids, embeddings, metadata))
        print(f"Uploaded batch {(i // batch_size) + 1} / {(len(chunks) // batch_size) + 1}")

def main():
    print("--- Starting Knowledge Base Setup ---")
    try:
        # 1. Fetch Data
        chunks = fetch_and_chunk_ai_data()
        
        # 2. Setup Database
        index = setup_pinecone()
        
        # 3. Upload Data
        upload_to_pinecone(index, chunks)
        
        print("\nSUCCESS: Knowledge base setup complete!")
        
    except Exception as e:
        print(f"\nERROR: Setup failed: {e}")

if __name__ == "__main__":
    main()
