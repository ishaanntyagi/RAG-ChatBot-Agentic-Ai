import langchain
import os    #OS to save the chromaDB to the working Directory etc
from langchain.text_splitter import CharacterTextSplitter #Breaks Text Documents into smaller Chunks
from langchain.embeddings import HuggingFaceEmbeddings    
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

#the in data_folder user will be storing the PDFs 
DATA_FOLDER = "data"            

#in the CHROMA_DB_FOLDER , here ChromaDB Will be Storing the Vector Database
CHROMA_DB_FOLDER = "chroma_db"

#Defining a Function to Read all the Pdf Files From a Folder and Converting them into langchain objects
def load_pdfs(folder):
    docs = []          #Empty List to Store Documents
    for filename in os.listdir(folder):   #Loops Through Every File in the Folder
        if filename.endswith(".pdf"):  #Adds the file to Loader 
            loader = PyPDFLoader(os.path.join(folder,filename))
    return docs


#Defining a Function that takes a list of documents and splits them into chunks 
def split_documents(documents):  
    splitter = CharacterTextSplitter(
        chunk_size = 1000, #only 1000 character Per chunk
        chunk_overlap=200 #Each Chunk has a Remembering Context power of 200 chars 
                          #Hence Helps to Maintain Memory 
    )
    return splitter.split_documents(documents)




#Now We have to turn the Chunks to Embeddings Hence Will be using the Hugging Face Transformer here

def embed_chunks(chunks):#This is the function
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #minLm Model Loader
    texts = [chunk.page_content for chunk in chunks] #creating a list of texts for each chunk.
    metadatas = [chunk.metadata for chunk in chunks] #List of Meta Data for each chunk 
    
    vectordb = Chroma.from_texts(      #a chroma vectorDB is not setUp
        texts=texts,
        embedding=embedder,
        persist_directory=CHROMA_DB_FOLDER, 
        metadatas=metadatas
    )
    vectordb.persist() #db saved to disk
    return vectordb #return the db objects 



#Main block of The file 
if __name__ == "__main__":        
    print("Loading PDFs...")   #simple Print Statement 
    docs = load_pdfs(DATA_FOLDER)  #Loads The pdfs
    print(f"Loaded {len(docs)} documents from PDFs.") 
    print("Splitting documents into chunks...") 
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    print("Embedding chunks and storing in ChromaDB...")
    vectordb = embed_chunks(chunks)
    print("PDFs embedded and stored! Ready for retrieval.")
    
     

            