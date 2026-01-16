import os
from dotenv import load_dotenv
from pinecone import Pinecone  # new SDK

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import PromptTemplate

load_dotenv(override=True)

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-knowledge-rag")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-flash-latest"

def get_rag_chain():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing in environment variables.")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing in environment variables.")

    # ----------------------------
    # Initialize Pinecone client
    # ----------------------------
    pc = Pinecone(api_key=pinecone_api_key)

    # Make sure index exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please run setup first.")

    # Connect to Pinecone index
    vector_index = pc.Index(PINECONE_INDEX_NAME)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # ----------------------------
    # Correct: do NOT pass 'environment' to PineconeVectorStore
    # ----------------------------
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Setup LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=google_api_key,
        temperature=0
    )

    # Strict prompt template
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

    qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    return qa_chain, vectorstore

