import langchain
import os    #OS to save the chromaDB to the working Directory etc
from langchain.text_splitter import CharacterTextSplitter #Breaks Text Documents into smaller Chunks
from langchain.embeddings import HuggingFaceEmbeddings    
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader # Added for .docx support

#the in data_folder user will be storing the PDFs and DOCX files 
DATA_FOLDER = "data"            

#in the CHROMA_DB_FOLDER , here ChromaDB Will be Storing the Vector Database
CHROMA_DB_FOLDER = "chroma_db"

#Defining a Function to Read all the Pdf and Docx Files From a Folder and Converting them into langchain objects
def load_docs(folder):
    docs = []          #Empty List to Store Documents
    print("Looking for document files in:", folder)
    for filename in os.listdir(folder):   #Loops Through Every File in the Folder
        print("Found file:", filename)
        if filename.lower().endswith(".pdf"):  #Case-insensitive check for PDF
            print("Loading PDF:", filename)
            try:
                loader = PyPDFLoader(os.path.join(folder,filename))
                pdf_docs = loader.load()
                print("Pages loaded from PDF:", len(pdf_docs))
                docs.extend(pdf_docs)
            except Exception as e:
                print("Error loading PDF:", filename, str(e))
        elif filename.lower().endswith(".docx"):  #Case-insensitive check for DOCX
            print("Loading DOCX:", filename)
            try:
                loader = Docx2txtLoader(os.path.join(folder,filename))
                docx_docs = loader.load()
                print("Pages loaded from DOCX:", len(docx_docs))
                docs.extend(docx_docs)
            except Exception as e:
                print("Error loading DOCX:", filename, str(e))
    print("Total documents loaded:", len(docs))
    return docs


#Defining a Function that takes a list of documents and splits them into chunks 
def split_documents(documents):  
    splitter = CharacterTextSplitter(
        chunk_size = 1000, #only 1000 character Per chunk
        chunk_overlap=200 #Each Chunk has a Remembering Context power of 200 chars 
                          #Hence Helps to Maintain Memory 
    )
    chunks = splitter.split_documents(documents)
    print("Total chunks created:", len(chunks))
    return chunks




#Now We have to turn the Chunks to Embeddings Hence Will be using the Hugging Face Transformer here

def embed_chunks(chunks):#This is the function
    if not chunks:
        print("No chunks to embed! Skipping embedding step.")
        return None
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #minLm Model Loader
    texts = [chunk.page_content for chunk in chunks] #creating a list of texts for each chunk.
    metadatas = [chunk.metadata for chunk in chunks] #List of Meta Data for each chunk 
    print("Embedding", len(texts), "chunks...")
    vectordb = Chroma.from_texts(      #a chroma vectorDB is not SetUp.
        texts=texts,
        embedding=embedder,
        persist_directory=CHROMA_DB_FOLDER, 
        metadatas=metadatas
    )
    vectordb.persist()  #db saved to disk
    return vectordb     #return the db objects


# Lets Work on the Vector Db Part Now , From The Second File That needs To Write on tHe disk 


import os 
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma


#Loading The Chroma Vector Embeddings 



def load_vectordb():
    embedder = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=CHROMA_DB_FOLDER,
        embedding_function=embedder              #The Embeddings Model Here is Loaded and Then
        )
    return vectordb




def retrieve_chunks(query, vectordb, k=3):
    results = vectordb.similarity_search(query,k=k)
    return results


# def main():
#     vectordb = load_vectordb()
#     print("Document Search Bot ~Ready!")
#     print("Type your question below. Type ~exit to quit.\n")
    
#     while True:
#         user_query = input("You: ")
#         if user_query.lower() == "exit":
#             print("Exiting chatbot. Goodbye!")
#             break
        
#         results = retrieve_chunks(user_query, vectordb)
#         print("\nTop relevant chunks from your documents:")
#         for i, doc in enumerate(results):
#             print("\n""-----------------------------------------------------------------------------------" "\n")
#             print(f"\nResult {i+1}:")
#             print(doc.page_content[:300])  #Provide Only the First 300 Chars.
#             print("Metadata:", doc.metadata)
#         print("\n-----------------------------------------------------------------\n")


#Except For tha Main BLock Considering Here , we will be Mentioning it At the End of The file 
# For now Lets Consider the File 3 for now 


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