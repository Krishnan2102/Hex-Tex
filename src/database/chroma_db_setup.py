import os
import uuid
import re
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

CHROMA_DIR = "./chroma_db" 
COLLECTION_NAME = "pdf_documents" 
DEFAULT_TEXT_FILE = "output.txt"

# Initialize Chroma Client
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Use a standard embedding function
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  
)

def read_extracted_text(file_path: str) -> str:
    try:
        # Added errors='ignore' to handle messy PDF extractions
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            print(f"Successfully read {len(text)} characters from {file_path}")
            return text
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def smart_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    # Split by form feed character (common in PDF exports for page breaks)
    pages = text.split('\x0c')
    chunks = []
    chunk_id = 0
    
    for page_num, page_text in enumerate(pages, 1):
        if not page_text.strip():
            continue

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', page_text)
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding this paragraph exceeds chunk_size
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'page': page_num,
                        'chunk_index': chunk_id,
                        'chunk_size': len(current_chunk)
                    }
                })
                chunk_id += 1

                # Create overlap: take the last 'overlap' characters of the current chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add the final remaining chunk of the page
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'page': page_num,
                    'chunk_index': chunk_id,
                    'chunk_size': len(current_chunk)
                }
            })
            chunk_id += 1

    print(f"Created {len(chunks)} chunks from the document")
    return chunks

def get_or_create_collection(collection_name: str = COLLECTION_NAME):
    try:
        # Check if it exists
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        print(f"Loaded existing collection: {collection_name}")
    except Exception:
        # Create it if it doesn't
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={
                "description": f"Vector store for {collection_name}",
                "document_type": "pdf"
            }
        )
        print(f"Created new collection: {collection_name}")
    return collection

def add_chunks_to_db(collection, chunks: List[Dict[str, Any]], pdf_source: str = None):
    if not chunks:
        return []

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Ensure source is in metadata without wiping other keys
    for meta in metadatas:
        meta['source'] = pdf_source or "unknown"

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(chunks)} chunks to collection '{collection.name}'")
    return ids

def query_database(collection, query: str, n_results: int = 5):
    return collection.query(
        query_texts=[query],
        n_results=n_results
    )

def print_query_results(results):
    print("\n" + "="*80)
    print("QUERY RESULTS")
    print("="*80)

    if not results or not results['documents'] or not results['documents'][0]:
        print("No results found")
        return

    # Loop through results
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # Lower distance = Higher similarity
        relevance = 1 - (distance / 2) # Normalized for L2 distance
        
        print(f"\n--- Result {i+1} (Relevance Score: {relevance:.3f}) ---")
        print(f" Page: {metadata.get('page', 'N/A')} | Source: {metadata.get('source', 'N/A')}")
        print(f" Text: {doc[:300]}..." if len(doc) > 300 else f" Text: {doc}")
        print("-" * 40)

def process_pdf_text(text_file: str = DEFAULT_TEXT_FILE, collection_name: str = COLLECTION_NAME):
    if not os.path.exists(text_file):
        print(f"File {text_file} does not exist. Please run your PDF extractor first.")
        return None

    document_text = read_extracted_text(text_file)
    if not document_text:
        return None
    
    chunks = smart_chunk_text(document_text)
    collection = get_or_create_collection(collection_name)
    
    pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
    add_chunks_to_db(collection, chunks, pdf_name)
    
    return collection

def main():
    collection = process_pdf_text(DEFAULT_TEXT_FILE, COLLECTION_NAME)
    if collection:
        print(f"\n Setup complete! Database: {CHROMA_DIR}")

def interactive_mode():
    try:
        collections = client.list_collections()
        if not collections:
            print("No collections found. Please run the script without --interactive first.")
            return
        
        # Use the first available collection
        target_name = collections[0].name
        collection = get_or_create_collection(target_name)
        
        print(f"Entering interactive mode for collection: {target_name}")
        while True:
            query = input("\nYour question (or 'quit'): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                results = query_database(collection, query)
                print_query_results(results)
    except Exception as e:
        print(f"An error occurred in interactive mode: {e}")

if __name__ == "__main__":
    import sys
    # Handle direct interactive flag
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
        response = input("\n Enter interactive mode? (y/n): ").lower()
        if response.startswith('y'):
            interactive_mode()

def get_relevant_context(query: str, n_results: int = 3):
    collection = get_or_create_collection()
    results = query_database(collection, query, n_results=n_results)
    
    formatted_results = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            })
    return formatted_results