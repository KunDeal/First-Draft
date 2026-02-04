import os
import logging
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base"
VECTOR_DB_DIR = "vector_db"

def cleanup_indexed_files():
    print("🧹 Starting cleanup of already indexed files...")
    
    if not os.path.exists(VECTOR_DB_DIR):
        print("❌ Vector DB not found. Nothing to clean.")
        return

    # Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection = client.get_collection(name="legal_rag")
    except Exception as e:
        print(f"❌ Error connecting to DB: {e}")
        return

    # Get all indexed filenames
    print("🔍 Fetching indexed files list...")
    try:
        # Get only metadatas to find filenames
        result = collection.get(include=["metadatas"])
        indexed_files = set()
        
        if result["metadatas"]:
            for meta in result["metadatas"]:
                if meta and "filename" in meta:
                    indexed_files.add(meta["filename"])
        
        print(f"📋 Found {len(indexed_files)} unique files in the database.")
        
    except Exception as e:
        print(f"❌ Error fetching metadata: {e}")
        return

    # Check source directory
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print("❌ Knowledge base directory not found.")
        return

    deleted_count = 0
    kept_count = 0
    
    files = os.listdir(KNOWLEDGE_BASE_DIR)
    print(f"📂 Checking {len(files)} files in {KNOWLEDGE_BASE_DIR}...")

    for filename in files:
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        
        if not os.path.isfile(filepath):
            continue
            
        if filename in indexed_files:
            try:
                os.remove(filepath)
                print(f"🗑️ Deleted already indexed file: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Could not delete {filename}: {e}")
        else:
            kept_count += 1

    print("-" * 30)
    print(f"✅ Cleanup complete.")
    print(f"🚫 Deleted: {deleted_count}")
    print(f"💾 Remaining: {kept_count}")

if __name__ == "__main__":
    cleanup_indexed_files()
