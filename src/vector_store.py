from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL, VECTOR_STORE_PATH
import json
import logging

logging.basicConfig(level=logging.INFO)

def build_vector_store():
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    logging.info(f"Building embeddings with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    logging.info("Creating FAISS index...")
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    logging.info(f"Saving index to {VECTOR_STORE_PATH}...")
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)  # added flag

if __name__ == "__main__":
    build_vector_store()