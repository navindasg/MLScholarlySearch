from langchain_chroma import Chroma
import os

def clear_vector_database():
    """
    Clear all data from the Chroma vector database using the updated langchain_chroma package.
    """
    persist_directory = "./vector_db"
    if not os.path.exists(persist_directory):
        print(f"Vector database directory {persist_directory} does not exist.")
        return
    
    print(f"Connecting to Chroma database at {persist_directory}...")
    vectorstore = Chroma(persist_directory=persist_directory)
    
    count_before = vectorstore._collection.count()
    print(f"Current number of vectors: {count_before}")
    
    if count_before == 0:
        print("Database is already empty.")
        return
    
    all_ids = vectorstore._collection.get()["ids"]
    
    print("Deleting all vectors...")
    vectorstore.delete(ids=all_ids)
    
    count_after = vectorstore._collection.count()
    print(f"Vectors after deletion: {count_after}")
    
    if count_after == 0:
        print("Database successfully cleared.")
    else:
        print(f"Warning: {count_after} vectors remain in the database.")

if __name__ == "__main__":
    clear_vector_database()
