import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import tiktoken
import json
import logging
import nltk
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_METHOD, SEMANTIC_THRESHOLD, EMBEDDING_MODEL

# ----- Ensure NLTK resources are available -----
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab resource...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt resource...")
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize

logging.basicConfig(level=logging.INFO)

tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    return len(tokenizer.encode(text))

# ----- Load embedding model ONCE globally -----
# We'll load it only if we're using semantic chunking
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None and CHUNKING_METHOD == "semantic":
        logging.info(f"Loading embedding model for semantic chunking: {EMBEDDING_MODEL}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

# ----- Semantic chunking (uses the cached model) -----
def semantic_chunking(text, threshold=SEMANTIC_THRESHOLD):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    model = get_embedding_model()
    if model is None:
        # Fallback to recursive if model not loaded
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    embeddings = model.encode(sentences, convert_to_tensor=False)

    # Compute similarities between consecutive sentences
    sims = []
    for i in range(len(sentences)-1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        sims.append(sim)

    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        if sims[i-1] >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    chunks.append(" ".join(current_chunk))
    return chunks

# ----- PDF loading with OCR fallback (unchanged) -----
def load_pdf_with_ocr(file_path):
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((page_num, text))
                logging.info(f"Page {page_num}: extracted with pdfplumber ({len(text)} chars)")
            else:
                logging.info(f"Page {page_num}: empty, trying OCR...")
                ocr_text = ocr_page(file_path, page_num)
                if ocr_text and ocr_text.strip():
                    pages.append((page_num, ocr_text))
                    logging.info(f"Page {page_num}: OCR extracted ({len(ocr_text)} chars)")
                else:
                    logging.warning(f"Page {page_num}: no text.")
    return pages

def ocr_page(file_path, page_num):
    try:
        images = convert_from_path(file_path, first_page=page_num, last_page=page_num, dpi=300)
        if not images:
            return ""
        image = images[0]
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        logging.error(f"OCR failed for page {page_num}: {e}")
        return ""

# ----- Main chunking dispatcher -----
def chunk_documents(pages):
    all_chunks = []
    # Pre-load the model once if using semantic chunking
    if CHUNKING_METHOD == "semantic":
        get_embedding_model()  # triggers loading

    for page_num, text in pages:
        if CHUNKING_METHOD == "semantic":
            try:
                chunks = semantic_chunking(text)
                # Re-join and enforce token limits using recursive splitter
                combined = "\n\n".join(chunks)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=tiktoken_len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                final_chunks = splitter.split_text(combined)
            except Exception as e:
                logging.warning(f"Semantic chunking failed for page {page_num}, falling back to recursive: {e}")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=tiktoken_len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                final_chunks = splitter.split_text(text)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=tiktoken_len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            final_chunks = splitter.split_text(text)

        for idx, chunk in enumerate(final_chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {"page": page_num, "chunk_id": idx}
            })
    return all_chunks

if __name__ == "__main__":
    logging.info("Loading PDF with OCR fallback...")
    pages = load_pdf_with_ocr(PDF_PATH)
    logging.info(f"Extracted text from {len(pages)} pages.")
    chunks = chunk_documents(pages)
    logging.info(f"Created {len(chunks)} chunks.")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logging.info("Chunks saved to chunks.json")