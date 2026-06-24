import json
import logging
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from src.vector_store import load_vector_store
from src.config import TOP_K, FINAL_K, RERANK_MODEL, GROQ_API_KEY, GROQ_MODEL, NUM_QUERIES

logging.basicConfig(level=logging.INFO)

# Load cross‑encoder once (cached)
reranker = CrossEncoder(RERANK_MODEL)

# ----- Synonym map for financial/business terms -----
SYNONYM_MAP = {
    "income": ["turnover", "revenue", "earnings", "total income"],
    "revenue": ["turnover", "income", "sales", "top line"],
    "turnover": ["income", "revenue", "sales"],
    "profit": ["earnings", "net income", "profit after tax"],
    "loss": ["net loss", "deficit"],
    "subsidiary": ["subsidiaries", "wholly owned", "step-down"],
    "ceo": ["chief executive officer", "managing director"],
    "cfo": ["chief financial officer"],
    "board": ["directors", "board of directors"],
    "risk": ["risks", "exposure", "uncertainty"],
    "growth": ["growth rate", "increase", "expansion"],
}

def apply_synonyms(queries: list) -> list:
    """Generate additional query variants by replacing terms with synonyms."""
    expanded = []
    for q in queries:
        expanded.append(q)  # keep original
        lower_q = q.lower()
        for word, syns in SYNONYM_MAP.items():
            if word in lower_q:
                for syn in syns:
                    # Only add if synonym not already in the query
                    if syn not in lower_q:
                        new_q = q.replace(word, syn)
                        expanded.append(new_q)
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for q in expanded:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:10]  # limit to 10 variants

# ----- End synonym map -----

def get_base_retriever():
    """Return the hybrid ensemble retriever (BM25 + semantic) with MMR."""
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = TOP_K

    vector_store = load_vector_store()
    semantic = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 2, "lambda_mult": 0.5}
    )
    ensemble = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.6, 0.4]
    )
    return ensemble

def expand_query(query: str) -> list[str]:
    """
    Generate alternative phrasings using Groq, then add synonym variants.
    """
    llm = ChatGroq(
        temperature=0.0,
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL
    )
    prompt = ChatPromptTemplate.from_template(
        f"You are a helpful assistant. Given a user question, generate {NUM_QUERIES} alternative ways to ask the same question. "
        "Return only the questions, one per line, numbered 1,2,3..."
        "\nOriginal question: {query}\nAlternative questions:"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    content = response.content if hasattr(response, 'content') else str(response)
    # Parse lines
    lines = content.strip().split('\n')
    alt_queries = []
    for line in lines:
        cleaned = line.strip()
        if cleaned and cleaned[0].isdigit():
            cleaned = cleaned.split('.', 1)[-1].strip()
            cleaned = cleaned.split(')', 1)[-1].strip()
        if cleaned:
            alt_queries.append(cleaned)
    alt_queries = alt_queries[:NUM_QUERIES]

    # Combine original + LLM alternatives + synonyms
    all_queries = [query] + alt_queries
    all_queries = apply_synonyms(all_queries)
    return all_queries

def rerank_documents(query: str, docs: list) -> list:
    """Rerank documents with cross‑encoder and return top FINAL_K."""
    if not docs:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:FINAL_K]]

def get_retriever(query: str):
    """
    Full retrieval pipeline:
    1. Expand query into multiple variants (LLM + synonyms).
    2. Retrieve for each variant using base retriever.
    3. Merge unique documents (by content).
    4. Rerank and return top FINAL_K.
    """
    base_retriever = get_base_retriever()
    alt_queries = expand_query(query)
    logging.info(f"Expanded queries: {alt_queries}")

    all_docs = []
    seen_texts = set()
    for q in alt_queries:
        docs = base_retriever.get_relevant_documents(q)
        for doc in docs:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                all_docs.append(doc)

    reranked = rerank_documents(query, all_docs)
    return reranked