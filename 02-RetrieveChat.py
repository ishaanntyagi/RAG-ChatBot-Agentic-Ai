import os 
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma


CHROMA_DB_FOLDER = "chroma_db"
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


def main():
    vectordb = load_vectordb()
    print("Document Search Bot ~Ready!")
    print("Type your question below. Type ~exit to quit.\n")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        
        results = retrieve_chunks(user_query, vectordb)
        print("\nTop relevant chunks from your documents:")
        for i, doc in enumerate(results):
            print("\n""-----------------------------------------------------------------------------------" "\n")
            print(f"\nResult {i+1}:")
            print(doc.page_content[:300])  #Provide Only the First 300 Chars.
            print("Metadata:", doc.metadata)
        print("\n-----------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    