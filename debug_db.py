print("Importing chromadb...")
import chromadb
print("chromadb imported successfully.")
import os
import ingest

# Configuration
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "legal_rag"

def debug_chroma():
    print("Starting debug_chroma function...")
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"Error: Directory '{VECTOR_DB_DIR}' not found.")
        return

    try:
        print(f"Connecting to ChromaDB at {VECTOR_DB_DIR}...")
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        
        # List all collections
        collections = client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
        
        print(f"Getting collection '{COLLECTION_NAME}'...")
        # Get collection
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # 1. Total Count
        total_count = collection.count()
        print(f"--- ChromaDB Diagnostic ---")
        print(f"Total count of chunks in collection '{COLLECTION_NAME}': {total_count}")
        
        # 2. Embedding Function
        print(f"Embedding model configured in code: {ingest.EMBEDDING_MODEL_NAME}")
        
        # Check if collection has an embedding function attached
        # (Though we know ingest.py adds embeddings manually)
        print(f"Collection embedding function: {getattr(collection, '_embedding_function', 'Manual/None')}")
        
        # 3. Content and Metadata of first 3 chunks
        if total_count > 0:
            results = collection.get(limit=3, include=["documents", "metadatas"])
            
            print(f"\n--- First 3 Chunks ---")
            for i in range(len(results["ids"])):
                print(f"\nChunk ID: {results['ids'][i]}")
                print(f"Metadata: {results['metadatas'][i]}")
                content_preview = results['documents'][i][:200].replace('\n', ' ')
                print(f"Content Preview: {content_preview}...")
                
                # Verify parent_id
                if "parent_id" in results['metadatas'][i]:
                    print(f"✅ parent_id present: {results['metadatas'][i]['parent_id']}")
                else:
                    print(f"❌ parent_id MISSING!")
        else:
            print("No data found in the collection.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    debug_chroma()
