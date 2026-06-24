import streamlit as st
from src.rag_chain import build_rag_chain
from src.retriever import get_retriever

# Page config
st.set_page_config(
    page_title="Swiggy RAG Assistant",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fc8019;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e8f0fe;
        border-left: 4px solid #fc8019;
    }
    .source-box {
        background-color: #fafafa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #fc8019;
    }
    .source-page {
        font-weight: 600;
        color: #fc8019;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_chain():
    return build_rag_chain()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = load_chain()

# Sidebar
with st.sidebar:
    st.image("https://logos-world.net/wp-content/uploads/2020/12/Swiggy-Logo.png", width=200)
    st.markdown("## 📄 About")
    st.markdown(
        """
        This assistant answers questions **strictly** from the 
        **Swiggy Annual Report 2023‑24**.
        
        - ✅ No hallucination – answers are grounded in the document.
        - 📚 Sources are shown with page numbers (click to expand).
        - 🔍 Uses hybrid retrieval + reranking for accuracy.
        
        **Example questions:**
        - What was Swiggy's total income?
        - Who is the CEO?
        - How many subsidiaries does Swiggy have?
        """
    )
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main area
st.markdown('<div class="main-header">🍔 Swiggy RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask any question about Swiggy\'s FY2023-24 annual report</div>', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"]["answer"])
            if "sources" in message["content"] and message["content"]["sources"]:
                # Sources are collapsed by default
                with st.expander("📚 Sources", expanded=False):
                    for i, (page, snippet) in enumerate(message["content"]["sources"], 1):
                        st.markdown(f"**Source {i}** – Page **{page}**")
                        st.caption(snippet[:300] + "..." if len(snippet) > 300 else snippet)
                        st.divider()
        else:
            st.markdown(message["content"])

# Input
if prompt := st.chat_input("Type your question here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chain(prompt)
                answer = response.content if hasattr(response, "content") else str(response)

                # Retrieve sources (top 3)
                docs = get_retriever(prompt)
                sources = [(doc.metadata.get("page", "Unknown"), doc.page_content) for doc in docs[:3]]

                # Display answer
                st.markdown(answer)

                # Display sources – collapsed by default
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for i, (page, snippet) in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** – Page **{page}**")
                            st.caption(snippet[:300] + "..." if len(snippet) > 300 else snippet)
                            st.divider()

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": {"answer": answer, "sources": sources},
                    }
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": {"answer": f"Sorry, I encountered an error: {e}", "sources": []},
                    }
                )