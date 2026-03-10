#they ain't believe in us, we the best music!

import os
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import re
from typing import List, Dict, Any
import json

#config
CHROMA_DIR = "./chroma_db" #database storage location
COLLECTION_NAME ="pdf_documents" #general collection name for now, will figure out how ts works later
DEFAULT_TEXT_FILE = "output.txt"

#initialize chroma client
client = chromadb.PersistentClient(
    path=CROMA_DIR,
    settings=Setiings(anonymized_telemetry=False)
)

#sentence transformers for embedding[runs locally]
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  
)

#text processing

def read_extracted_text(./pdf_documents/ :str) ->str:
    try:
        with open(./pdf_documents/, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f"Successfully read {len(text)} characters from {./pdf_documents/}")
                  return text
except FileNotFoundError:
print(f"Error: {./pdf_documents/} not found")
      return ""
except Exception as e:
    print(f"Error reading file: {e}")
    return ""

    def smart_chunk_test(text: str, chunk_size: int =1000, overlap: int = 200) ->List[Dict[str,Any]]:

        #Split by pages first
    pages = text.split('\x0c')

    chunks = []
    chunk_id = 0
    
    for page_num, page_text in enumerate(pages, 1):
    if not page_text.strip():
        continue

#split page in paragraph
paragraphs = re.split(r'\n\s*\n', page_text)
current_chunk = ""
for para in paragraphs:
    para = para.strip()
    if not para:
        continue

    #if paragraph exceeds chunk size, then we save the current chunk or something
    if len(current_chunk) + len(para) > chunk_size and current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'metadata': {
                'page': page_num,
                'chunk_index': chunk_id,
                'chunk_size': len(current_chunk),
                'source': os.path.basename(DEFAULT_TEXT_FILE).replace('.txt', '.pdf')
            }
        })
        chunk_id += 1

        #start new chunk with overlap from previous chunk
        words = current_chunk.split()
        overlap_text = ' '.join(words[-50:]) if len(words) > 50 else
            current_chunk
        current_chunk = overlap_text + "\n\n" + para
        else:

            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

                #the last chunk of the page

                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'page':page_num,
                            'chunk_index': chunk_id,
                            'chunk_size': len(current_chunk),
                            'source': os.path.basename(DEFAULT_TEXT_FILE).replace('.txt','.pdf')
                            }
                        }
                                  )
                    chunk_id += 1
                    print(f"created {len(chunks)} chunks from the document")
                    return chunks

                def chunk_text_simple(text:str, chunk_size: int = 1000, overlap: int =200) ->List[Dict[str, Any]]:

                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'start_char': i,
                            'end_char': i + len(chunk_text),
                            'chunk_index': len(chunks),
                            'source': os.path.basename(DEFAULT_TEXT_FILE).replace('.txt', '.pdf')
                            }
                        }
                                  )
                    print(f"Created {len(chunks)} chunks using simple method")
                    return chunks

                


                # CHROMADB OPERATIONSS

                def get_or_create_collection(collection_name:str = COLLECTION_NAME):
                    try:
                        collection = client.get_collection{
                                name=collection_name,
                                embedding_function=embedding_func
                                )
                        print(f"loaded existing collection: {collection_name}"}
                    except valueerror:
                         #this collection doesn't exist so we can create it
                         collection = client.create_collection(
                                 name=collection_name,
                                 embedding_function=embedding_func,
                                 metadata={
                                     "description": f"vector store for {collection_name}",
                                     "created_with": "chroma_db_setup.py",
                                     "document_type": "pdf"
                                     }
                                     }
                                     print(f" created new collection: {collection_name}")

                                     return collection

                                     def add_chunks_to_db(collection, chunks: List[Dict[str, Any]], pdf_source: str = None):

                                     #preparing data for chromadb

                                     ids =[str(uuid.uuid4()) for _ in chunks]
                                     documents = [chunks['text'] for chunk in chunks]
                                     metadata['source'] = pdf_source

                                     #adding to database
                                     collection.add(
                                             documents=documents,
                                             metadatas=metadatas,
                                             ids=ids
                                             )

                                     print(f"Added {len(chunks)} chunks to collection '{collection.name}'")
                                     return ids


                                 def query_database(collection, query: str, n_results: int = 5):

                                     results = collection.query(
                                             query_text=[query],
                                             n_results=n_results
                                             )
                                     return results

                                 
                                 def print_query_results(results):
                                     print("\n" + "="*80)
                                     print("QUERY RESULTS")
                                     print("="*80)

                                     if not results['documents'][0]:
                                         print("no results found")
                                         return

                                     for i, (doc, metadata, distance) in enumerate(zip(
                                             results['documents']0],
                                             results['metadatas'][0],
                                             results['distances'][0]
                                         )):
                                             relevance = 1 - distance  # Convert distance to similarity score
                                             print(f"\n--- Result {i+1} (Relevance: {relevance:.3f}) ---")
                                             if 'page' in metadata:
                                             print(f" Page: {metadata['page']}")
                                                if 'source' in metadata:
                                                print(f"Source: {metadata['source']}")
                                                
                                                print(f" Text: {doc[:200]}..." if len(doc) > 200 else f" Text: {doc}")
                                                print("-"*40)


def process_pdf_text(text_file: str = DEFAULT_TEXT_FILE, collection_name: str = COLLECTION_NAME):

print(f"\n Processing PDF text from: {text_file}")
    print(f" Using collection: {collection_name}")
    
    # Reading the extracted text
    print("\n Step 1: Reading document...")
    document_text = read_extracted_text(text_file)
    if not document_text:
        print(" No text to process. Exiting.")
        return None
    
    #Chunk the text
    print("\n✂️ Step 2: Chunking document...")
    chunks = smart_chunk_text(document_text)
    
    # Preview first chunk
    if chunks:
        print(f"\n Preview of first chunk:")
        print("-" * 40)
        preview = chunks[0]['text'][:300]
        print(preview + "..." if len(chunks[0]['text']) > 300 else chunks[0]['text'])
        print("-" * 40)
    
    #create collection
    print("\n🗄️ Step 3: Setting up database collection...")
    collection = get_or_create_collection(collection_name)
    
    #add chunks to database
    print("\n Step 4: Adding chunks to database...")
    pdf_name = os.path.basename(text_file).replace('.txt', '.pdf')
    ids = add_chunks_to_db(collection, chunks, pdf_name)
    
    #show stats
    print("\n Collection Statistics:")
    print(f"   - Total chunks in collection: {collection.count()}")
    if chunks:
        print(f"   - Pages processed: {len(set([c['metadata'].get('page', 0) for c in chunks if 'page' in c['metadata']]))}")
    
    return collection

#query func for fastapi

def get_relevant_context(query: str, collection_name: str = COLLECTION_NAME, n_results: int = 5) -> List[Dict[str, Any]]:
       try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results for easy use
        contexts = []
        if results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                contexts.append({
                    'text': doc,
                    'metadata': metadata,
                    'relevance_score': 1 - distance  # Convert to similarity
                })
        
        return contexts
    except Exception as e:
        print(f"Error querying database: {e}")
        return []

def list_all_collections():
    """List all available collections in the database"""
    collections = client.list_collections()
    print("\n Available Collections:")
    if collections:
        for coll in collections:
            count = coll.count()
            print(f"  • {coll.name} ({count} documents)")
    else:
        print("  No collections found.")
    return collections


#main function

def main():
    print("\n" + "="*60)
    print("PDF VECTOR DATABASE SETUP")
    print("="*60)
    
    # show available collections
    list_all_collections()
    
    # processing the PDF text
    collection = process_pdf_text(DEFAULT_TEXT_FILE, COLLECTION_NAME)
    
    if collection:
        # testing the database with sample queries
        print("\n Testing database with sample queries...")
        
        test_queries = [
            "What is this document about?",
            "What are the main topics?",
            "Tell me about the key points"
        ]
        
        for query in test_queries:
            results = query_database(collection, query, n_results=2)
            print(f"\n Query: '{query}'")
            if results['documents'][0]:
                print(f"   Found: {results['documents'][0][0][:150]}...")
            else:
                print("   No results found")
        
        print(f"\n Setup complete!")
        print(f" Database stored in: {CHROMA_DIR}")
        print(f" Collection name: {COLLECTION_NAME}")
        print(f" Total documents in collection: {collection.count()}")
    
    print("\n" + "="*60)



#interactive mode part

def interactive_mode():
    print("\n" + "="*60)
    print(" INTERACTIVE QUERY MODE")
    print("="*60)
    
    # Shows available collections
    collections = list_all_collections()
    
    if not collections:
        print("\n No collections found. Please run the setup first.")
        return
    
    # Let user choose collection
    print("\nWhich collection would you like to query?")
    for i, coll in enumerate(collections, 1):
        print(f"  {i}. {coll.name} ({coll.count()} documents)")
    
    try:
        choice = int(input("\nEnter number (or 0 for default): ")) - 1
        if 0 <= choice < len(collections):
            collection_name = collections[choice].name
        else:
            collection_name = COLLECTION_NAME
    except:
        collection_name = COLLECTION_NAME
    
    collection = get_or_create_collection(collection_name)
    
    print(f"\n Querying collection: {collection_name}")
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if query:
            results = query_database(collection, query)
            print_query_results(results)

#script using

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Run in interactive mode
        interactive_mode()
    else:
        # Run the main setup
        main()
        
        # Ask if user wants to go to interactive mode
        response = input("\n Enter interactive query mode? (y/n): ").lower()
        if response.startswith('y'):
            interactive_mode()
