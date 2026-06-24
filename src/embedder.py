import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_and_persist_vector_store(chunks, persist_directory: str):
    """Generates embeddings and creates a FAISS vector store, then saves it."""
    print(f"Initializing HuggingFace embeddings model...")
    # Using a fast, free local model suitable for sentences
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"Creating FAISS index with {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Persisting vector store to {persist_directory}...")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    vector_store.save_local(persist_directory)
    
    print("Vector store successfully created and persisted.")
    return vector_store

def load_vector_store(persist_directory: str):
    """Loads a persisted FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # allow_dangerous_deserialization=True is required for FAISS when loading local pickles, which is safe here as we created it
    vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    return vector_store

if __name__ == "__main__":
    from ingest import load_and_chunk_pdf
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "swiggy_annual_report.pdf")
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "vector_store")
    
    chunks = load_and_chunk_pdf(pdf_path)
    if chunks:
        create_and_persist_vector_store(chunks, persist_dir)
