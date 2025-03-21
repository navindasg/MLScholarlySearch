import os
from pdf_parser import PDFParser
from embeddings import create_embeddings_from_directory
from table_handler import TableHandler

def process_all_documents(input_dir: str = "./PDFinput", 
                        output_dir: str = "./parser_output",
                        vector_db_dir: str = "./vector_db"):
    """Process all documents through the complete pipeline."""
    
    print("\n=== Step 1: Parsing PDFs ===")
    # Initialize parser
    parser = PDFParser(output_dir=output_dir)
    
    # Process each PDF
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            print(f"\nProcessing {pdf_path}...")
            parser.process_pdf(pdf_path)
    
    # Close parser resources
    parser.close()
    
    print("\n=== Step 2: Creating Embeddings ===")
    # Create embeddings from processed text files
    create_embeddings_from_directory(
        input_dir=output_dir,
        vector_db_dir=vector_db_dir,
        file_pattern="*.txt"
    )
    
    print("\n=== Pipeline Complete ===")
    print(f"- PDFs processed and saved to: {output_dir}")
    print(f"- Embeddings created in: {vector_db_dir}")
    print(f"- Tables and image descriptions stored in: tables.db")
    print("\nThe system is now ready for querying!")

if __name__ == "__main__":
    # Clear any existing processed files
    import shutil
    import sqlite3
    
    print("=== Cleaning Previous Data ===")
    
    # Clear vector store
    if os.path.exists("./vector_db"):
        shutil.rmtree("./vector_db")
        print("Cleared vector store")
    
    # Clear parser output
    if os.path.exists("./parser_output"):
        shutil.rmtree("./parser_output")
        print("Cleared parser output")
    
    # Clear tables database
    if os.path.exists("tables.db"):
        os.remove("tables.db")
        print("Cleared tables database")
    
    # Create necessary directories
    os.makedirs("./parser_output", exist_ok=True)
    os.makedirs("./vector_db", exist_ok=True)
    
    # Run the pipeline
    process_all_documents() 
