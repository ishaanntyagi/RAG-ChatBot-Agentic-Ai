# ChatRAG: RAG Chatbot & Summarizer via AI and NLP

---

## Screenshots


![File Structure and GitHub Clean-up](Screenshot%202025-07-24%20170506.png)

- Shows clean repo with all main scripts and no `chroma_db/` errors.

| Home Page / App UI | Summarization Example | Chatbot Example | Chat History Example |
|--------------------|----------------------|-----------------|---------------------|
| ![Home UI](Screenshot%202025-07-24%20170516.png) | ![Summarize](Screenshot%202025-07-24%20170915.png) | ![Chatbot](Screenshot%202025-07-24%20170945.png) | ![Chat History](Screenshot%202025-07-24%20171338.png) |

- **Home Page:** Choose to summarize or chat.
- **Summarize:** Upload PDF, get extractive/AI summary and keywords.
- **Chatbot:** Ask questions about your uploaded document, see top relevant sections and AI answers.
- **Chat History:** Review previous questions and answers.

---

## Features

- **Summarize PDF (and DOCX) documents**
  - Extractive summaries (using embeddings)
  - Generative summaries (using Ollama/LLM)
- **Chat with your documents** (RAG-style Q&A)
  - Upload documents, ask questions, get context-rich answers
- **Keyword extraction** and document chunking
- **Simple web UI** (Streamlit) and command-line interface
- **Beginner-friendly code** with comments and stepwise logic
- **No proprietary SaaS needed** (runs locally with Ollama and open models)
- **Modular code:** Each part (embedding, retrieval, answering) is in its own file

---

## Project Structure

```plaintext
.
├── App.py                # Main CLI app: RAG chatbot & summarizer functions
├── St-App.py             # Streamlit UI app for ChatRAG
├── 0.py                  # Example: LLM call via Ollama API (test)
├── 01-embedDocs.py       # Step 1: Load & embed documents, store in ChromaDB
├── 02-RetrieveChat.py    # Step 2: Retrieve relevant chunks from ChromaDB
├── 03-LLM-Answer.py      # Step 3: Generate answers using LLM and retrieved docs
├── data/                 # Folder for your PDF/DOCX files
├── chroma_db/            # Chroma vector DB folder (auto-created, should be gitignored)
└── README.md             # You are here!
```

---

## How it Works

1. **Document Embedding:**  
   You upload your PDF/DOCX files. The code splits them into chunks, creates embeddings using a HuggingFace model, and stores them in a Chroma vector database.

2. **Retrieval-Augmented Generation (RAG):**  
   When you ask a question, the chatbot finds the most relevant document chunks using vector similarity search.

3. **Answer Generation:**  
   The most relevant chunks, plus your question, are sent to a local LLM (via Ollama API) to generate a context-aware answer.  
   Optionally, you can ask the LLM to answer using its own knowledge.

4. **Summarization:**  
   Summarize your documents either using embedding-based extractive summaries or generative summaries via the LLM.

5. **User Interfaces:**  
   - **Command-line (App.py):** For those who like to run and test code directly.
   - **Streamlit UI (St-App.py):** For a modern, easy-to-use web interface.

---

## Tech Stack

- **Python 3.8+**
- [LangChain](https://github.com/langchain-ai/langchain) (embeddings, document loaders, chunking)
- [Ollama](https://ollama.com/) (local LLM API)
- [ChromaDB](https://www.trychroma.com/) (vector database)
- [Streamlit](https://streamlit.io/) (web interface)
- [PyPDF2](https://pypi.org/project/PyPDF2/) (PDF reading)
- [HuggingFace Sentence Transformers](https://www.sbert.net/) (embeddings)

---

## Notes & Tips

- Make sure to keep `chroma_db/` in your `.gitignore` to avoid large file errors in git.
- If you ever need to clear the database, just delete the `chroma_db/` folder.
- You can change the embedding model or LLM model in the code (check `EMBED_MODEL` and `OLLAMA_MODEL`).
- The code is heavily commented for learning and experimentation!

---

## Credits

- Built using open-source libraries by the AI and Python community.
- LLMs served locally via [Ollama](https://ollama.com/).
- Inspired by [LangChain](https://python.langchain.com/) RAG and chatbot demos.

---

Happy learning and building with Generative AI!  
**Questions or feedback?** Open an issue or start a discussion.
