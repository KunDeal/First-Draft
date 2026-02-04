import chromadb
import os

VECTOR_DB_DIR = "vector_db"

def check_all_files():
    if not os.path.exists(VECTOR_DB_DIR):
        print("❌ Папка базы данных не найдена.")
        return

    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection = client.get_collection(name="legal_rag")
        
        # Get all metadata (limit to 1000 just in case, but we expect ~20)
        data = collection.get(include=["metadatas"])
        
        unique_files = set()
        total_chunks = len(data["ids"])
        
        if data["metadatas"]:
            for meta in data["metadatas"]:
                if meta and "filename" in meta:
                    unique_files.add(meta["filename"])
        
        print(f"📊 Статистика базы данных:")
        print(f"   - Всего фрагментов (chunks): {total_chunks}")
        print(f"   - Всего уникальных файлов: {len(unique_files)}")
        print("\n📂 Список проиндексированных файлов:")
        for i, filename in enumerate(sorted(unique_files), 1):
            print(f"   {i}. {filename}")
            
    except Exception as e:
        print(f"❌ Ошибка при чтении базы: {e}")

if __name__ == "__main__":
    check_all_files()
