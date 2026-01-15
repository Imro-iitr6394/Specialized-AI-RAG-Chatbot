import os
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_and_chunk_ai_data():
    # Topics related to Artificial Intelligence
    topics = [
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Neural network",
        "Natural language processing",
        "Computer vision",
        "Generative artificial intelligence",
        "Large language model",
        "Reinforcement learning",
        "Ethics of artificial intelligence"
    ]
    
    documents = []
    
    print(f"Fetching data for {len(topics)} topics from Wikipedia...")
    
    for topic in topics:
        try:
            print(f"Fetching: {topic}")
            page = wikipedia.page(topic, auto_suggest=False)
            content = page.content
            documents.append({
                "text": content,
                "metadata": {
                    "source": page.url,
                    "title": page.title,
                    "topic": "Artificial Intelligence"
                }
            })
        except Exception as e:
            print(f"Error fetching {topic}: {e}")
            
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for i, text in enumerate(split_texts):
            chunks.append({
                "text": text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_id": i
                }
            })
            
    print(f"Total chunks created: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    chunks = fetch_and_chunk_ai_data()
    # Save a few chunks to a file for preview
    with open("chunks_preview.txt", "w", encoding="utf-8") as f:
        for i in range(min(5, len(chunks))):
            f.write(f"--- Chunk {i} ---\n")
            f.write(f"Source: {chunks[i]['metadata']['source']}\n")
            f.write(chunks[i]['text'][:200] + "...\n\n")
