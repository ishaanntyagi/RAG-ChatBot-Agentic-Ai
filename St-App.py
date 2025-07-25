import streamlit as st
from App import (
    summarize_text_embeddings,
    extract_keywords,
    ollama_generate,
    load_vectordb,
    retrieve_chunks,
    load_docs,
    split_documents,
    embed_chunks
)
import tempfile
import os

st.set_page_config(page_title="ChatRAG AI Assistant", page_icon="🤖", layout="wide")

# Sidebar navigation with improved logic: if on Home, show choices as buttons, not navigation
def sidebar_navigation():
    st.sidebar.image("https://em-content.zobj.net/source/microsoft-teams/363/robot_1f916.png", width=80)
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Summarize", "Chatbot", "Chat History"])
    return page

# Session state for chat history and summary type
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "summary_type" not in st.session_state:
    st.session_state["summary_type"] = "Normal (Embeddings)"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

def set_page(page):
    st.session_state["current_page"] = page

# Sidebar logic
page = sidebar_navigation()

# --- HOME PAGE ---
if page == "Home":
    st.title("🤖 ChatRAG AI Assistant")
    st.markdown("""
    Welcome! What would you like to do?
    """)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Summarize a Document", use_container_width=True):
            set_page("Summarize")
            st.session_state["current_page"] = "Summarize"
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st._rerun()
    with col2:
        if st.button("Chat with Your Documents", use_container_width=True):
            set_page("Chatbot")
            st.session_state["current_page"] = "Chatbot"
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st._rerun()
    st.info("You can always switch pages from the sidebar.")

# --- SUMMARIZE PAGE ---
elif page == "Summarize":
    st.header("📄 Document Summarizer")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Use BUTTONS for summary type selection (not radio!)
    st.markdown("**Choose summary type:**")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Normal (Embeddings)", key="emb_btn", use_container_width=True):
            st.session_state["summary_type"] = "Normal (Embeddings)"
    with c2:
        if st.button("AI (Ollama)", key="ai_btn", use_container_width=True):
            st.session_state["summary_type"] = "AI (Ollama)"
    st.write(f"Selected: {st.session_state['summary_type']}")

    # Button to generate summary (Image 3 style: only generate on click)
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if not text or len(text.strip()) < 50:
            st.error("No usable text found in PDF.")
        else:
            if st.button("Generate Summary", key="gen_summary", use_container_width=True):
                if st.session_state["summary_type"] == "AI (Ollama)":
                    with st.spinner("Generating AI summary..."):
                        prompt = f"Summarize the following PDF text in 5 sentences:\n{text}"
                        ai_summary = ollama_generate(prompt)
                        st.markdown("<div style='background-color: #234c1d; color: white; padding:10px; border-radius:8px; margin-bottom:6px;'>AI Summary (Ollama)</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='background-color:#e8f0fe;padding:18px;border-radius:10px;'><span style='font-size:1.1em;'><b>ChatRAG:</b> {ai_summary}</span></div>", unsafe_allow_html=True)
                else:
                    with st.spinner("Generating embedding-based summary..."):
                        summary = summarize_text_embeddings(text, top_n=5)
                        st.markdown("<div style='background-color: #234c1d; color: white; padding:10px; border-radius:8px; margin-bottom:6px;'>Extractive Summary (Embeddings)</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='background-color:#e8ffe8;padding:18px;border-radius:10px;'><span style='font-size:1.1em;'><b>ChatRAG:</b> {summary}</span></div>", unsafe_allow_html=True)

                # Show keywords
                keywords = extract_keywords(text)
                st.markdown("**Top Keywords:**")
                st.markdown(f"<div style='background-color:#fffbe8;padding:10px;border-radius:8px;'>"
                            f"<span style='font-size:1.1em;'>{', '.join(keywords)}</span></div>", unsafe_allow_html=True)
    else:
        st.info("Upload your PDF to summarize.")

# --- CHATBOT PAGE ---
elif page == "Chatbot":
    st.header("💬 Chat with Your Documents")
    uploaded_file = st.file_uploader("Upload a PDF file to chat with", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        docs = load_docs(os.path.dirname(pdf_path))
        chunks = split_documents(docs)
        vectordb = embed_chunks(chunks)
        st.markdown("<div style='background-color:#234c1d; color: white; padding:10px; border-radius:8px;'>Document loaded! You can now ask questions below.</div>", unsafe_allow_html=True)

        query = st.text_input("Ask a question about your document:", key="chat_input")
        # Use Button to generate answer (like Image 3)
        if st.button("Get Answer", use_container_width=True):
            with st.spinner("Searching your document..."):
                results = retrieve_chunks(query, vectordb)
            if not results:
                st.error("No relevant information found in your document for this question.")
                st.session_state.chat_history.append({"user": query, "bot": "No relevant information found in your documents for this question."})
            else:
                st.markdown("**Top relevant sections:**")
                for i, doc in enumerate(results):
                    st.expander(f"Section {i+1}").write(doc.page_content[:500] if doc.page_content else "[No content]")
                with st.spinner("Thinking..."):
                    prompt = f"""
                    You are an assistant helping users answer questions using the provided document context.

                    Instructions:
                    - Use the provided context as much as possible to answer the question.
                    - If you can't find the answer, try to reason with the information given.
                    - If the answer is truly not present, say: "The answer is not available in the provided context."
                    - Be clear and concise.

                    Context:
                    {' '.join([doc.page_content for doc in results])}

                    Question: {query}

                    Answer:
                    """
                    bot_answer = ollama_generate(prompt)
                    st.session_state.chat_history.append({"user": query, "bot": bot_answer})
                    st.markdown("<div style='background-color:#e6f7ff;padding:16px;border-radius:8px;margin-top:8px;'><b>ChatRAG:</b> " +
                                bot_answer + "</div>", unsafe_allow_html=True)
                # Optionally ask for general AI answer
                if st.button("Get AI answer using general knowledge", use_container_width=True):
                    ai_prompt = f"Answer the following question using your own knowledge:\n\nQuestion: {query}\nAnswer:"
                    with st.spinner("Getting AI answer..."):
                        ai_answer = ollama_generate(ai_prompt)
                    st.markdown("<div style='background-color:#fff0f0;padding:16px;border-radius:8px;'><b>ChatRAG (General):</b> " +
                                ai_answer + "</div>", unsafe_allow_html=True)
                    st.session_state.chat_history.append({"user": f"AI knowledge: {query}", "bot": ai_answer})
    else:
        st.info("Upload your PDF to chat with.")

# --- CHAT HISTORY PAGE ---
elif page == "Chat History":
    st.header("📚 Chat History")
    if st.session_state.chat_history:
        for i, turn in enumerate(st.session_state.chat_history):
            st.markdown(f"**User {i+1}:** {turn['user']}")
            st.markdown(f"<div style='background-color:#e6f7ff;padding:10px;border-radius:8px;'><b>ChatRAG:</b> {turn['bot']}</div>", unsafe_allow_html=True)
            st.markdown("---")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")
    else:
        st.info("No chat history yet.")

# --- FOOTER ---
st.markdown("""
---
<span style='font-size:1em;'>| Powered by Ollama & LangChain</span>
""", unsafe_allow_html=True)