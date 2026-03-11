import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PartnerA_Ingestion:
    def __init__(self):
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\x0c", "\n\n", "\n", " "]
        )

    def process_for_db(self, pdf_path):
        """
        Extracts and chunks text specifically for Partner B's 
        add_chunks_to_db function.
        """
        try:
            doc = fitz.open(pdf_path)
            all_chunks = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
               
                page_chunks = self.splitter.split_text(text)
                
                for i, chunk_text in enumerate(page_chunks):
                   
                    all_chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'page': page_num + 1,
                            'chunk_index': i,
                            'source': os.path.basename(pdf_path)
                        }
                    })
            return all_chunks
        except Exception as e:
            print(f"Error: {e}")
            return []

