import os, json, traceback
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import read_docx, get_doc_id, EMBEDDING_MODEL_NAME

out = {"steps": []}
try:
    out["steps"].append(f"model={EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    out["steps"].append("model_loaded")
    c = chromadb.PersistentClient(path="vector_db")
    col = c.get_or_create_collection("legal_rag")
    files = [f for f in os.listdir("knowledge_base") if f.endswith(".docx")]
    out["files"] = files
    if not files:
        out["error"] = "no_files"
    else:
        p = os.path.join("knowledge_base", files[0])
        text = read_docx(p) or ""
        out["text_len"] = len(text)
        chunk = text[:1000] or "empty"
        doc_id = get_doc_id(files[0])
        pid = f"{doc_id}_0"
        meta = {
            "parent_id": pid,
            "doc_id": doc_id,
            "filename": files[0],
            "page_content": chunk,
            "part_index": 0,
            "total_parts": 1,
            "prev_parent_id": "",
            "next_parent_id": "",
        }
        emb = model.encode(chunk).tolist()
        col.add(ids=[pid + "_c0"], embeddings=[emb], metadatas=[meta], documents=[chunk])
        out["added"] = True
        out["count"] = col.count()
except Exception as e:
    out["exception"] = str(e)
    out["traceback"] = traceback.format_exc()

with open("quick_add_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("DONE")
