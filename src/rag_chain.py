from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from src.config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE
from src.retriever import get_retriever
import logging

logging.basicConfig(level=logging.INFO)

def format_docs(docs):
    return "\n\n".join(
        f"Page {doc.metadata.get('page', 'Unknown')}:\n{doc.page_content}"
        for doc in docs
    )

def build_rag_chain():
    llm = ChatGroq(
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL
    )

    template = """
    You are an AI assistant that answers questions based SOLELY on the provided context.
    Do not use any external knowledge or your own prior information.
    
    - If the answer is clearly found in the context, provide a concise answer and mention the page number(s).
    - If the context contains relevant numbers but does NOT explicitly state the answer (e.g., you need to compute a growth rate), you may perform simple arithmetic (e.g., subtraction, division) using numbers from the context, but do NOT add any external data.
    - If the answer cannot be found or is not sufficiently covered in the context, say "I don't have information about that" and explain why (e.g., "The context mentions total income but does not provide a growth percentage.").
    
    Always cite the page number(s) from the context that support your answer, even if you say "I don't know".
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def rag_pipeline(query):
        docs = get_retriever(query)
        context = format_docs(docs)
        chain = prompt | llm
        return chain.invoke({"context": context, "question": query})

    return rag_pipeline