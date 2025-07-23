from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import os

CHROMA_DB_FOLDER = "chroma_db"

#------------------------------------------------------------------------------

def load_vectordb():
    embedder = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory  = CHROMA_DB_FOLDER, 
        embedding_function = embedder
    )
    return vectordb


#------------------------------------------------------------------------------


def retrieve_chunks(query, vectordb, k=3):
    results = vectordb.similarity_search(query,k=k)
    return results


#------------------------------------------------------------------------------


import requests
OLLAMA_URL   = ("http://localhost:11434/api/generate")
OLLAMA_MODEL = "llama3.2"

def ollama_generate(prompt, model=OLLAMA_MODEL, url=OLLAMA_URL):
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "stream" : False
    }
       
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")


#------------------------------------------------------------------------------


def generate_answer(query,results):
    context = "\n\n".join([doc.page_content for doc in results])
    
    
    prompt = f"""
    You are an expert assistant helping users answer questions using the provided document context.

    Instructions:
    - Only use the provided context to answer the question.
    - If the answer is not in the context, reply exactly: "The answer is not available in the provided context."
    - Do not make up information.
    - Be clear and concise.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    
    answer = ollama_generate(prompt)
    
    # If answer not found, ask user if they want the AI to generate from its own knowledge
    # if "The answer is not available in the provided context." in answer:
    #     print("Bot: The answer is not available in your documents.")
    #     user_choice = input("Would you like the AI to try answering from its own knowledge? (yes/no): ")
    #     if user_choice.strip().lower() == "yes":
    #         new_prompt = f"Answer the following question using your own knowledge:\n\nQuestion: {query}\nAnswer:"
    #         answer = ollama_generate(new_prompt)
    #         print(f"Bot: {answer}\n")
    #     else:
    #         print("Bot: Okay, not generating an answer outside your documents.\n")
    # else:
    #     print(f"Bot: {answer}\n")
        
                
    user_choice = input("Want AI's answer too? (yes/no): ")
    if user_choice.strip().lower() == "yes":
        ai_prompt = f"Answer the following question using your own knowledge:\n\nQuestion: {query}\nAnswer:"
        ai_answer = ollama_generate(ai_prompt)
        print(f"Bot (AI knowledge): {ai_answer}\n")
    else:
        print("Bot: Okay, not generating an answer outside your documents.\n")

        
        
#------------------------------------------------------------------------------

def chat():
    vectordb = load_vectordb()
    "\n"
    "\n"
    "\n"
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    
    print("Document-powered Chatbot with Ollama Llama 3.2 ready! Type your question below. Type 'exit' to quit.\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        results = retrieve_chunks(user_query, vectordb)
        print("\nTop relevant chunks from your documents:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content[:300])  # Show first 300 chars for preview
            print("Metadata:", doc.metadata)
        print("\nGenerating answer using Ollama Llama 3.2...\n")
        generate_answer(user_query, results)
        print("---\n")

#------------------------------------------------------------------------------

if __name__ == "__main__":
    chat()
   

    
    
    
    