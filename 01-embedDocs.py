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



#Main block of The file: 
if __name__ == "__main__":        
    print("Loading documents...")   #simple Print Statement 
    docs = load_docs(DATA_FOLDER)  #Loads The pdfs and docx files
    print(f"Loaded {len(docs)} documents from files.") 
    print("Splitting documents into chunks...") 
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    print("Embedding chunks and storing in ChromaDB...")
    vectordb = embed_chunks(chunks)
    if vectordb is not None:
        print("Documents embedded and stored! Ready for retrieval.")
    else:
        print("No data to embed. Please check your document content.")