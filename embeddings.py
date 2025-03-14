from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

def process_documents():
    """
    Process text files from ./parser_output directory:
    1. Load all text files
    2. Chunk them using RecursiveCharacterTextSplitter
    3. Generate embeddings using nomic-embed-text from Ollama
    4. Store in Chroma vector database
    """
    input_directory = "./parser_output"
    persist_directory = "./vector_db"
    if not os.path.exists(input_directory):
        print(f"Error: Directory {input_directory} does not exist.")
        return
    print(f"Loading documents from {input_directory}...")
    loader = DirectoryLoader(input_directory, glob="**/*.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    if len(documents) == 0:
        print(f"No text files found in {input_directory}.")
        return
    print("Initializing Ollama embeddings with nomic-embed-text model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    print(f"Creating vector store at {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"Vector store created and persisted at {persist_directory}")
    print(f"Total vectors: {vectorstore._collection.count()}")
    return vectorstore

if __name__ == "__main__":
    process_documents()
