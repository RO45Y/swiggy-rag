markdown
# 🍔 Swiggy Annual Report RAG Assistant

A production‑grade Retrieval-Augmented Generation (RAG) application that answers natural‑language questions **strictly** from the Swiggy Annual Report 2023‑24, using Groq AI, local embeddings, hybrid retrieval with MMR, semantic chunking, and a chat‑friendly UI.

Built as part of an AI/ML assignment – no hallucinations, only context‑grounded answers with source citations.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🧠 Architecture](#-architecture)
- [⚙️ Setup & Installation](#-setup--installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Building the Vector Store](#building-the-vector-store)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#-configuration)
- [🧪 Evaluation](#-evaluation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📄 License](#-license)

---

## ✨ Features

- ✅ **Accurate RAG pipeline** – hybrid retrieval (BM25 + semantic) with **MMR** (Maximum Marginal Relevance) for diversity, and a cross‑encoder reranker for precision.
- 🚀 **Fast generation** with Groq AI (free tier, <1s responses).
- 🧠 **Local embeddings** – uses `BAAI/bge-large-en-v1.5` (robust to OCR noise, no API cost).
- 📚 **Sources shown** – page numbers and context snippets, collapsed by default for a clean UI.
- 💬 **Chat‑style UI** with persistent history, side‑bar info, and one‑click clear.
- 🔒 **Strict grounding** – prompt forces the model to answer only from the retrieved context, preventing hallucination. Supports simple arithmetic (e.g., growth percentages) when numbers are present.
- 🧹 **Intelligent chunking** – **semantic chunking** groups coherent sentences together, with a fallback to token‑aware recursive splitting. OCR fallback (Tesseract) handles scanned pages.
- 🔍 **Query expansion** – generates multiple paraphrased questions using Groq, plus **synonym substitution** for financial and business terms to boost recall.
- 📊 **Tunable retrieval** – all parameters (chunk size, overlap, top‑k, reranker, ensemble weights, MMR lambda) are centralised in `config.py`.
- 🧪 **Built‑in evaluation** – `evaluate.py` measures **Recall@5** and **MRR** against a custom test set, enabling quantitative performance tracking.

---

## 🧠 Architecture

The system follows a modular, production‑ready pipeline:

1. **Ingestion** (`src/ingest.py`):  
   - Extracts text via `pdfplumber` (for selectable text) and falls back to **Tesseract OCR** for scanned pages.  
   - Applies **semantic chunking**: splits text into sentences, computes cosine similarity between consecutive sentences, and merges them into coherent chunks while respecting token limits.  
   - Falls back to `RecursiveCharacterTextSplitter` if semantic chunking fails.

2. **Embedding & Vector Store** (`src/vector_store.py`):  
   - Generates embeddings using `HuggingFaceEmbeddings` with `BAAI/bge-large-en-v1.5` (configurable).  
   - Builds and persists a **FAISS** index for fast similarity search.

3. **Retrieval** (`src/retriever.py`):  
   - **Hybrid retriever** combining BM25 (keyword) and FAISS (semantic) with tunable weights (default `[0.6, 0.4]`).  
   - **MMR** (Maximum Marginal Relevance) to diversify retrieved chunks.  
   - **Query expansion**:  
     - Groq‑generated paraphrases (via LLM).  
     - **Synonym substitution** for key financial terms (e.g., income↔turnover, subsidiary↔subsidiaries).  
   - **Cross‑encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re‑order the best candidates.

4. **Generation** (`src/rag_chain.py`):  
   - Uses a strict prompt that forces the LLM to answer **solely** from the provided context.  
   - Instructs the model to say “I don’t have information” if the answer is not found, and to cite page numbers.  
   - Allows simple arithmetic (e.g., growth rate calculation) if the numbers are present.  
   - Invokes **Groq** (`llama-3.3-70b-versatile`) for fast, accurate responses.

5. **UI** (`app.py`):  
   - Streamlit chat interface with sidebar, chat history, and expandable sources.  
   - Displays retrieved context with page numbers for transparency.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Homebrew (macOS) or equivalent package manager (for Tesseract & Poppler)
- Groq API key (free tier) – get it from [console.groq.com](https://console.groq.com)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ro45y/swiggy-rag.git
   cd swiggy-rag
Create a virtual environment

bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# or
venv\Scripts\activate         # Windows
Install dependencies

bash
pip install -r requirements.txt
Install system dependencies for OCR (macOS)

bash
brew install tesseract poppler
For Linux/Windows, refer to the official Tesseract and Poppler installation guides.

Set up environment variables

bash
cp .env.example .env
Edit .env and add your Groq API key:

env
GROQ_API_KEY=your_groq_api_key_here
Place the Swiggy Annual Report

Download the PDF and place it as data/swiggy_annual_report.pdf.

You can adjust the path in src/config.py (PDF_PATH).

Building the Vector Store
Run the ingestion and vector store build once (this may take a few minutes, especially with OCR and semantic chunking):

bash
python -m src.ingest          # Creates chunks.json (uses semantic chunking)
python -m src.vector_store    # Builds FAISS index in vector_store/
Note: If you change chunk size, embedding model, or chunking method, delete chunks.json and the vector_store/ folder before rebuilding.

🚀 Usage
Start the Streamlit app:

bash
streamlit run app.py
Open http://localhost:8501 and start asking questions.

Example questions
What was Swiggy's consolidated total income for FY2023‑24?

Who is the Managing Director & Group CEO?

How many subsidiaries does Swiggy have?

What was the total income of Scootsy Logistics?

Does the company have any material exposure to equity price risk?

What was the percentage increase in consolidated total income?

The assistant displays the answer and, beneath it, collapsed source cards (click to expand) showing the page number and a snippet of the retrieved context.

📁 Project Structure
text
swiggy-rag/
├── app.py                    # Streamlit UI (chat interface)
├── data/
│   └── swiggy_annual_report.pdf
├── src/
│   ├── __init__.py
│   ├── config.py             # All configuration parameters
│   ├── ingest.py             # PDF ingestion, OCR fallback, semantic chunking
│   ├── vector_store.py       # Embeddings + FAISS index building
│   ├── retriever.py          # Hybrid retrieval, MMR, query expansion, reranking
│   ├── rag_chain.py          # Prompt, LLM invocation, answer formatting
│   └── utils.py              # Helper functions (optional)
├── vector_store/             # FAISS index (auto‑generated)
├── chunks.json               # Chunks cache (auto‑generated)
├── eval_questions.json       # Example evaluation test set
├── evaluate.py               # Evaluation script (Recall@5, MRR)
├── .env                      # API keys (ignored by git)
├── .env.example              # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
⚙️ Configuration
All tunable parameters are in src/config.py:

Parameter	Description	Default
CHUNK_SIZE	Number of tokens per chunk	800
CHUNK_OVERLAP	Overlap between chunks (tokens)	200
CHUNKING_METHOD	Chunking strategy: "semantic" or "recursive"	"semantic"
SEMANTIC_THRESHOLD	Similarity threshold for sentence merging	0.65
EMBEDDING_MODEL	Sentence‑transformer model for embeddings	BAAI/bge-large-en-v1.5
TOP_K	Number of candidates retrieved before reranking	18
FINAL_K	Number of chunks passed to the LLM after reranking	5
ENSEMBLE_WEIGHTS	Weights for BM25 vs semantic retrieval	[0.6, 0.4]
MMR_LAMBDA	MMR diversity parameter (0 = max diversity, 1 = max relevance)	0.5
GROQ_MODEL	Groq LLM model	llama-3.3-70b-versatile
TEMPERATURE	LLM temperature (0 = deterministic)	0.0
NUM_QUERIES	Number of paraphrased questions to generate	3
RERANK_MODEL	Cross‑encoder for reranking	cross-encoder/ms-marco-MiniLM-L-12-v2
Adjust these to improve retrieval quality or adapt to a different document.

🧪 Evaluation
To measure retrieval performance, we provide a small evaluation framework.

Create a test set – eval_questions.json should contain questions and expected page numbers, e.g.:

json
[
  {"question": "What was Swiggy's consolidated total income?", "pages": [6]},
  {"question": "What was the total income of Scootsy Logistics?", "pages": [29]}
]
Run the evaluation script:

bash
python evaluate.py
Output example:

text
Recall@5: 85.00%
MRR: 0.9123
Recall@5 – fraction of questions where the correct page appears in the top‑5 retrieved chunks.

MRR – Mean Reciprocal Rank – indicates how early the correct page is found on average.

This allows you to quantitatively tune parameters (e.g., TOP_K, ensemble weights, MMR lambda) for best performance.

🙏 Acknowledgements
Groq – for the fast, free LLM API.

LangChain – for the RAG orchestration.

Sentence‑Transformers & FAISS – for local embeddings and vector search.

Streamlit – for the interactive UI.

Tesseract & Poppler – for OCR support.

scikit-learn & NLTK – for semantic chunking and sentence tokenization.

📄 License
This project is for educational purposes only. Use of the Swiggy Annual Report is subject to Swiggy’s copyright.