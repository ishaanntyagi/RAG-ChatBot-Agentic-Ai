import os

from langchain.text_splitter import RecursiveCharacterTextSplitter #To Split The txt and Chars
from langchain_community.embeddings import HuggingFaceBgeEmbeddings #To Convert To Embeddings 
from langchain_community.vectorstores import Chroma #Chroma Vector DB
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader #To load The pdf and Docx 

import requests
import string
import numpy as np
import re

# Setting Up The basics , PreRequisite
DATA_FOLDER = "data"
CHROMA_DB_FOLDER = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" #This is The Embedding Model
OLLAMA_URL = "http://localhost:11434/api/generate" #Local Api url , to Send The User Query with Context to Ollama
OLLAMA_MODEL = "llama3.2"

chat_history = [] #Empty List named as ChatHistory So that later the Context and the Response Can be Appended to This List.

custom_stopwords = set([
    'the', 'is', 'in', 'and', 'to', 'of', 'a', 'from', 'for', 'on', 'with', 'at', 'by', 'an', 'as', 'be', 'has', 'have', 'are', 'was', 'were', 'or', 'that', 'this', 'but', 'their', 'it'
])  #These are the stopwords , That are needed to be Removed.


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import string
def simple_tokenize(text):
    
    for p in string.punctuation:
        text = text.replace(p, '')  #this will Remove The punctuation !!
    return text.lower().split() #This Will Turn Them To lower Case Letters.


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def extract_keywords(text, top_k=5):
    # Make all words lowercase and split by spaces (tokenize)
    words = text.lower().split()
    # List to store the important words (not stopwords)
    important = []
    for word in words:
        if word not in custom_stopwords and word.isalpha():
            important.append(word)
    # Count how often each word appears
    freq = {}
    for word in important:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    # Sort words by frequency (biggest first)
    sorted_words = sorted(freq, key=freq.get, reverse=True)
    # Return top_k words
    return sorted_words[:top_k]

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def split_sentences(text):
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    return sentences #here All the , Sentences are being Splitted. 
    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def summarize_text_embeddings(text, top_n=5):
    sentences = split_sentences(text)  #calling to Split Sentences
    
    if len(sentences) <= top_n:            
        return '. '.join(sentences) + '.'
    
    embedder = HuggingFaceBgeEmbeddings(model_name = EMBED_MODEL)     #Embeddings are being Created.
    embeddings = embedder.embed_documents(sentences)                   
    
    
    embeddings = np.array(embeddings) #Converting The Embedding into Numpy array
    doc_embedding = np.mean(embeddings, axis=0) #Calculates the average 
    
    #Measuring The Similarities 
    similarities = np.dot(embeddings, doc_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(doc_embedding))
    
    
    top_indices = similarities.argsort()[-top_n:][::-1]                #Finds the indices of The top_n
    selected_sentences = [sentences[i] for i in sorted(top_indices)]   
    summary = '. '.join(selected_sentences) + '.'
    return summary
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def load_docs(folder): 
    docs = []       #To add The Docs to This 
    print("ChatRAG: Looking for documents in:", folder) #Finding the Documents , if there is any.
    
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):  #Check if the folder name is ending With The .pdf Format
            loader = PyPDFLoader(os.path.join(folder, filename)) #Using PyPdfLoader 
            docs.extend(loader.load()) #Loading the Pdf 
            
            
        elif filename.lower().endswith(".docx"):   #Check if the format is .docx , as we have functionality of Both Pdf and Docx 
            loader = Docx2txtLoader(os.path.join(folder, filename))
            docs.extend(loader.load())  #Loading the docx
            
    print("ChatRAG: Total documents loaded:", len(docs))
    return docs   #Return Docx and Priting the TOtal number of docs 


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def split_documents(documents): #There might be Many Docs And we have to Split them and Break Them into smaller Chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    #The langchainCharacterSplitter Is used Here.
    #Divides the Document text into , chunkSize of 800 , and chunk overlap of 100 , to retain the memeory of Previous Chunk , hence Context Aware 
    
    chunks = splitter.split_documents(documents)  #Performing the Splitting 
    print("ChatRAG: Total chunks created:", len(chunks)) #Providing the split and via chatRag
    return chunks


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def embed_chunks(chunks): #taking all the chunks 
    embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)   #using the Hugging Face Embeddings Mlm model
    texts = [chunk.page_content for chunk in chunks] # Loop To get the Data 
    metadatas = [chunk.metadata for chunk in chunks] # Loop to get the Meta Data Associated with the Chunks or Doc Data.
    vectordb = Chroma.from_texts(
        texts = texts,
        embedding = embedder,
        persist_directory = CHROMA_DB_FOLDER,    #Providing the Details about the Chroma Db Folder And the , embedder model and The meta data
        metadatas = metadatas
    )
    return vectordb


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def load_vectordb():
    embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)  #Same HuggingFace Model.
    vectordb = Chroma(  #Db used = Chroma
        persist_directory = CHROMA_DB_FOLDER,   #Here This Line will be helping it , Write to the folder defined for the chroma Part 
        embedding_function  = embedder          #Embedder Function/ Model Taken As parameter By the Vector Db Function
    )
    return vectordb


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def retrieve_chunks(query, vectordb, k=5):              #The Vector database to fetch the Embeds From its Sqlite Database , Query by the user , and k is the number of Chunks That will be Provided to the user 
    results = vectordb.similarity_search(query, k=k)    #All The chunks Fetched will be used and similarity 
                                                        #Will be found in Between them , And will alsobe compared to the Query (q)
    return results


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def ollama_generate(prompt, model=OLLAMA_MODEL, url=OLLAMA_URL):    #parameters , Such as prompt , ollamaURL ( 11432API URL), as well as the Ollma Model
    payload = {
                "model": model,        #The Model Defined above That is , Llama3.2 ~ 4Billion train Parameters via ollama 
                "prompt": prompt,      #To Initiate Ollama Request , and provide it the Prompt
                "stream": False        
                }
    
    response = requests.post(url, json=payload) # Fetch in the form of JSON Format
    result = response.json()
    return result.get("response", "") #Results



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def generate_answer(query, results):
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""
    You are an assistant helping users answer questions using the provided document context.

    Instructions:
    - Use the provided context as much as possible to answer the question.
    - If you can't find the answer, try to reason with the information given.
    - If the answer is truly not present, say: "The answer is not available in the provided context."
    - Be clear and concise.

    Context:
    {context}

    Question: {query}        

    Answer:
    """
    #This is the Prompt That is Provided to The LLM 
    
    answer = ollama_generate(prompt)    #Sent To the Ollama Api 
    print("ChatRAG: AI Answer based only on your documents:") #Reply
    print(answer)   #The answer --
    chat_history.append({"user": query, "bot": answer})  #The Chat will be added , To the Chat List Created By us Earlier.
    
    
    user_choice = input("Would you like an AI answer using general knowledge? Type 'yes' or 'no': ") #The Bots own Knowledge, Will be used If user has Provided his Choice 
    
    if user_choice.strip().lower() == "yes":
        ai_prompt = f"Answer the following question using your own knowledge:\n\nQuestion: {query}\nAnswer:"
        ai_answer = ollama_generate(ai_prompt)  #Ollama 
        print("AI Answer using its own knowledge:")
        print(ai_answer)
        chat_history.append({"user": f"AI knowledge: {query}", "bot": ai_answer})  #Chat Being appended
        
        
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
        


def summarize_pdf_flow():
    filename = input("ChatRAG: Enter PDF file name (from 'data' folder): ")  # Ask for the PDF name in the data folder
    pdf_path = os.path.join(DATA_FOLDER, filename)                           # Get full PDF path
    from PyPDF2 import PdfReader                                            # Import PDF reader
    reader = PdfReader(pdf_path)                                            # Read the PDF
    text = ""                                                               # Store all text here
    for page in reader.pages:                                               # Go through each page
        page_text = page.extract_text()                                     # Get text from the page
        if page_text:                                                       # If there is text
            text += page_text + " "                                         # Add it to main text
    if not text or len(text.strip()) < 50:                                  # If no useful text found
        print("No usable text found in PDF.")                               # Tell the user and stop
        return

    print("ChatRAG: How do you want to summarize your PDF?")                # Ask how to summarize
    print("         1: Normal (embedding-based, extractive summary)")       # Option 1: normal summary
    print("         2: Ollama/AI based summary")                            # Option 2: AI summary
    user_choice = input("Enter 1 or 2: ").strip()                          # Get user choice

    if user_choice == "2":                                                  # If AI summary
        prompt = f"Summarize the following PDF text in 5 sentences:\n{text}"# Build prompt for Ollama
        ai_summary = ollama_generate(prompt)                                # Get summary from Ollama
        print("\nChatRAG: PDF Summary (Ollama/AI based):")                  # Show AI summary
        print(ai_summary)
    else:                                                                   # If normal summary
        summary = summarize_text_embeddings(text, top_n=5)                  # Run normal summary
        print("\n ChatRAG: PDF Summary (Extractive, Embedding-based):")     # Show normal summary
        print(summary)

    keywords = extract_keywords(text)                                       # Get keywords
    print("\nChatRAG:  Top Keywords:")                                      # Show keywords
    print(", ".join(keywords))                                              # Print keywords
    print()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def chat():
    vectordb = load_vectordb()                                              # Load your vector database
    print("ChatRAG: Welcome to your Document-powered AI Chatbot!")          # Welcome the user
    print("         Type your question below and press Enter.")             # How to use
    print("         Type 'exit' to quit. Type 'history' to see previous chat.") # More options
    
    
    while True:                                                             # Keep chatting until exit
        user_query = input("User: ")                                        # Get user question
        if user_query.lower().strip() == "exit":                            # If user wants to quit
            print("Exiting chatbot. Have a great day!")                     # Say goodbye
            break                                                           # Stop the loop
        
        
        if user_query.lower().strip() == "history":                         # If user wants history
            print("Previous Chat History:")                                 # Show chat history
            for i, turn in enumerate(chat_history):                         # Go through history
                print(f"\nUser {i+1}: {turn['user']}\nAI: {turn['bot']}")   # Show each chat turn
            continue                                                        # Next question
        
        
        results = retrieve_chunks(user_query, vectordb)                     # Find relevant text chunks
        if not results:                                                     # If nothing found
            print("ChatRAG: No relevant information found in your documents for this question.")
            chat_history.append({"user": user_query, "bot": "No relevant information found in your documents for this question."}) # Save to history
            continue                                                        # Try again
        
        
        print("ChatRAG: Top relevant sections from your documents:")        # Show best matches
        for i, doc in enumerate(results):                                   # Go through results
            print(f"Result {i+1}:")                                         # Show result number
            print(doc.page_content[:300] if doc.page_content else "[No content]") # Show first 300 chars
        generate_answer(user_query, results)                                # Get answer from AI
        print("ChatRAG: -- End of this answer --\n")                        # Divider for answers
        
        
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def main():
    print("ChatRAG: Welcome! Please select a mode:")
    print("         1: Chat with your documents (RAG Chatbot)")
    print("         2: Summarize the full document (extractive, embedding-based)")
    print("         Exit: To Exit")
    choice = input("Enter 1, 2, or 3: ").strip()   #This is The main Function Calling all the Other Things that need to be Used within the code 

    
    if choice == "1":
        docs = load_docs(DATA_FOLDER)
        chunks = split_documents(docs)      #The Choice 1
        embed_chunks(chunks)
        chat()
        
    elif choice == "2":                     #The Choice 2
        summarize_pdf_flow()
    else:
        print("Exiting.")

if __name__ == "__main__":              #The main Function
    main()

    
