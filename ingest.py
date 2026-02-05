import os
import hashlib
import datetime
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import docx
import fitz  # PyMuPDF
import pandas as pd  # Excel support
from dotenv import load_dotenv
import time
import gc

# Load environment variables
load_dotenv()

# Configuration - Golden Settings for 8GB RAM
KNOWLEDGE_BASE_DIR = "knowledge_base"
VECTOR_DB_DIR = "vector_db"
LOGS_DIR = "logs"
CHUNK_SIZE = 800       # Reduced from 1500
CHUNK_OVERLAP = 200    # Reduced from 250
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# Setup logging - Minimal mode
def setup_logging():
    # Only log errors to file, keep console clean
    logging.basicConfig(
        filename=os.path.join(LOGS_DIR, "ingest.log"),
        level=logging.ERROR, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

if __name__ == "__main__":
    print("DEBUG: Script started")
    setup_logging()

def get_file_hash(filepath):
    """Calculate MD5 hash of a file to detect changes."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def read_docx(filepath):
    """Extract text from a .docx file."""
    try:
        doc = docx.Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        error_msg = f"Error reading DOCX {filepath}: {e}"
        print(error_msg)
        logging.error(error_msg)
        return None

def read_pdf(filepath):
    """Extract text from a .pdf file with page numbers."""
    try:
        doc = fitz.open(filepath)
        pages_content = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_content.append({"text": text, "page": page_num + 1})
        doc.close()
        return pages_content
    except Exception as e:
        error_msg = f"Error reading PDF {filepath}: {e}"
        print(error_msg)
        logging.error(error_msg)
        return None

def read_excel(filepath):
    """Extract text from a .xlsx file, converting each row to text."""
    try:
        # Read Excel file
        df = pd.read_excel(filepath)
        
        # Convert to string, handling NaNs
        text_content = []
        
        # Iterate over rows
        for index, row in df.iterrows():
            row_text = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):  # Skip empty cells
                    row_text.append(f"{col}: {val}")
            
            if row_text:
                # Join cells in a row with ", " and rows with newline
                text_content.append(" | ".join(row_text))
        
        return "\n".join(text_content)
    except Exception as e:
        error_msg = f"Error reading Excel {filepath}: {e}"
        print(error_msg)
        logging.error(error_msg)
        return None

def recursive_split_text(text, chunk_size, chunk_overlap):
    """Split text recursively respecting separators."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break
        
        # Try to find a split point (newline, space)
        split_point = -1
        # Prioritize separators in order: Paragraphs > Lines > Sentences > Words
        for separator in ["\n\n", "\n", ". ", " "]:
            last_sep = text.rfind(separator, start, end)
            if last_sep != -1 and last_sep > start:
                split_point = last_sep + len(separator)
                break
        
        if split_point == -1:
            split_point = end
        
        chunks.append(text[start:split_point])
        
        # Calculate next start
        next_start = split_point - chunk_overlap
        
        # Prevent infinite loops: ensure we always move forward
        if next_start <= start:
            next_start = split_point
            
        start = next_start
        
        if start < 0: start = 0
        
    return chunks

def ingest_documents():
    # Force garbage collection at start
    gc.collect()
    
    # Initialize ChromaDB
    print("DEBUG: Initializing ChromaDB")
    try:
        # ChromaDB client is lightweight
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection = chroma_client.get_or_create_collection(name="legal_rag")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB: {e}")
        return

    # Initialize Embedding Model on CPU
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        return

    # Optimization: existing hashes check
    existing_hashes = set()
    try:
        existing_data = collection.get(include=["metadatas"])
        if existing_data["metadatas"]:
            for meta in existing_data["metadatas"]:
                if meta and "file_hash" in meta:
                    existing_hashes.add(meta["file_hash"])
        
        del existing_data
        gc.collect()
    except Exception as e:
        logging.error(f"Error reading existing DB: {e}")

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        logging.error(f"Knowledge base directory not found created: {KNOWLEDGE_BASE_DIR}")
        return

    all_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if os.path.isfile(os.path.join(KNOWLEDGE_BASE_DIR, f))]
    total_files = len(all_files)
    
    print(f"Starting ingestion of {total_files} files in Stable/Low-RAM mode...")
    print(f"Settings: Chunk={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}, Batch=4")

    # Strict 1-by-1 Processing
    for i, filename in enumerate(all_files):
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        
        try:
            # 1. Check Hash
            file_hash = get_file_hash(filepath)
            if file_hash in existing_hashes:
                print(f"[{i+1}/{total_files}] Skipping {filename} (already indexed)")
                continue

            chunks_to_add = []
            metadatas_to_add = []
            ids_to_add = []
            
            # 2. Read and Chunk
            if filename.lower().endswith(".docx"):
                text = read_docx(filepath)
                if text is None:
                    continue # Error already printed
                if text:
                    raw_chunks = recursive_split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                    del text # Clear full text
                    
                    for idx, chunk in enumerate(raw_chunks):
                        chunks_to_add.append(chunk)
                        metadatas_to_add.append({
                            "filename": filename,
                            "page_number": 1,
                            "upload_date": datetime.datetime.now().isoformat(),
                            "file_hash": file_hash,
                            "chunk_index": idx
                        })
                        ids_to_add.append(f"{file_hash}_{idx}")
                    
                    del raw_chunks

            elif filename.lower().endswith(".pdf"):
                pages = read_pdf(filepath)
                if pages is None:
                    continue # Error already printed
                if pages:
                    chunk_counter = 0
                    for page_data in pages:
                        page_text = page_data["text"]
                        page_num = page_data["page"]
                        
                        raw_chunks = recursive_split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
                        
                        for chunk in raw_chunks:
                            chunks_to_add.append(chunk)
                            metadatas_to_add.append({
                                "filename": filename,
                                "page_number": page_num,
                                "upload_date": datetime.datetime.now().isoformat(),
                                "file_hash": file_hash,
                                "chunk_index": chunk_counter
                            })
                            ids_to_add.append(f"{file_hash}_{chunk_counter}")
                            chunk_counter += 1
                    
                    del pages

            elif filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
                text = read_excel(filepath)
                if text is None:
                    continue
                if text:
                    raw_chunks = recursive_split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                    del text
                    
                    for idx, chunk in enumerate(raw_chunks):
                        chunks_to_add.append(chunk)
                        metadatas_to_add.append({
                            "filename": filename,
                            "page_number": 1, # Excel treated as single page
                            "upload_date": datetime.datetime.now().isoformat(),
                            "file_hash": file_hash,
                            "chunk_index": idx
                        })
                        ids_to_add.append(f"{file_hash}_{idx}")
                    
                    del raw_chunks

            # 3. Embed and Add (Only if we have chunks)
            if chunks_to_add:
                # CRITICAL: Use batch_size=4 to prevent RAM spikes during embedding
                # Enable progress bar to show activity on large files
                embeddings = model.encode(chunks_to_add, batch_size=4, show_progress_bar=True).tolist()
                
                collection.add(
                    documents=chunks_to_add,
                    embeddings=embeddings,
                    metadatas=metadatas_to_add,
                    ids=ids_to_add
                )
                
                print(f"[{i+1}/{total_files}] Indexed {filename} ({len(chunks_to_add)} chunks)")
                
                # 4. Explicit cleanup
                del chunks_to_add
                del metadatas_to_add
                del ids_to_add
                del embeddings
            else:
                 print(f"[{i+1}/{total_files}] Skipped {filename} (no text found)")

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            print(f"[{i+1}/{total_files}] Error on {filename}: {e}")
        
        # 5. Garbage Collection after EVERY file
        gc.collect()
        time.sleep(0.1) # Brief pause to let system settle
        
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()
