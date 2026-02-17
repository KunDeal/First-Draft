import os
import hashlib
import re
import logging
import chromadb
import json
import shutil
from sentence_transformers import SentenceTransformer
import docx
import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
import gc

load_dotenv()

KNOWLEDGE_BASE_DIR = "knowledge_base"
VECTOR_DB_DIR = "vector_db"
DOC_STORE_DIR = "doc_store"
LOGS_DIR = "logs"
MODELS_DIR = "models"

PARENT_CHUNK_SIZE = 3000
CHILD_CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "legal_rag"

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "ingest.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def reset_databases():
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    os.makedirs(DOC_STORE_DIR, exist_ok=True)


def get_doc_id(filename):
    return hashlib.md5(filename.encode("utf-8")).hexdigest()


def read_docx(filepath):
    try:
        doc = docx.Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logging.error(f"Error reading DOCX {filepath}: {e}")
        return None


def read_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts)
    except Exception as e:
        logging.error(f"Error reading PDF {filepath}: {e}")
        return None


def read_excel(filepath):
    try:
        df = pd.read_excel(filepath)
        rows = []
        for _, row in df.iterrows():
            parts = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    parts.append(f"{col}: {value}")
            if parts:
                rows.append(" | ".join(parts))
        return "\n".join(rows)
    except Exception as e:
        logging.error(f"Error reading Excel {filepath}: {e}")
        return None


def recursive_split(text, chunk_size, overlap):
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end == text_len:
            chunks.append(text[start:])
            break
        split_point = -1
        search_start = max(start, end - int(chunk_size * 0.2))
        for sep in ["\n\n", "\n", ". ", " "]:
            candidate = text.rfind(sep, search_start, end)
            if candidate != -1:
                split_point = candidate + len(sep)
                break
        if split_point == -1:
            split_point = end
        chunks.append(text[start:split_point])
        next_start = split_point - overlap
        if next_start <= start:
            next_start = start + chunk_size - overlap
        start = max(start + 1, next_start)
    return chunks


def save_parent_to_store(doc_id, part_index, text, filename):
    parent_id = f"{doc_id}_{part_index}"
    file_path = os.path.join(DOC_STORE_DIR, f"{parent_id}.json")
    payload = {
        "parent_id": parent_id,
        "doc_id": doc_id,
        "part_index": part_index,
        "filename": filename,
        "text": text,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return parent_id


def sanitize_metadata(metadata):
    clean = {}
    for k, v in metadata.items():
        if v is None:
            if k.startswith("is_"):
                clean[k] = False
            elif "index" in k or "parts" in k:
                clean[k] = -1
            else:
                clean[k] = ""
        else:
            clean[k] = v
    return clean


def load_embedding_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_dir = os.path.join(MODELS_DIR, "e5-small")
    if not os.path.isdir(local_dir) or not os.listdir(local_dir):
        logging.error(f"Local embedding model not found or empty: {local_dir}")
        raise RuntimeError(f"Local embedding model not found: {local_dir}")
    model = SentenceTransformer(local_dir, device="cpu")
    return model


def is_gk4_filename(filename):
    lower = filename.lower()
    return "гражданский кодекс российской федерации" in lower and "часть четвертая" in lower


def split_gk4_into_articles(text):
    articles = []
    pattern = re.compile(r"(?m)^(Статья\s+(\d+[.\d]*)\..*?)\s*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return articles
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group(1)
        number = match.group(2)
        body = text[start:end].strip()
        articles.append((number, body))
    return articles


def split_gk4_article_children(article_text):
    lines = article_text.splitlines()
    chunks = []
    current = []
    paragraph_re = re.compile(r"^\s*(\d+(\.\d+)*)\.\s+")
    for line in lines:
        if paragraph_re.match(line):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    cleaned = [c for c in chunks if c]
    if not cleaned:
        cleaned = recursive_split(article_text, CHILD_CHUNK_SIZE, CHUNK_OVERLAP)
    return cleaned


def ingest_gk_only():
    logging.info("Starting GK4 ingestion into separate collection 'legal_gk'")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        logging.error(f"{KNOWLEDGE_BASE_DIR} not found")
        return

    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_or_create_collection(name="legal_gk")

    existing_files = set()
    try:
        existing_data = collection.get(include=["metadatas"])
        for meta in existing_data.get("metadatas", []):
            if meta and "filename" in meta:
                existing_files.add(meta["filename"])
    except Exception as e:
        logging.error(f"Error reading existing metadata for GK: {e}")

    model = load_embedding_model()

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if not is_gk4_filename(filename):
            continue

        if filename in existing_files:
            logging.info(f"Skipping already indexed GK4 file: {filename}")
            continue

        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        logging.info(f"Processing GK4 file: {filename}")

        text = None
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = read_pdf(filepath)
        elif lower.endswith(".docx"):
            text = read_docx(filepath)
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            text = read_excel(filepath)

        if not text:
            logging.error(f"No text extracted from {filename}, skipping")
            continue

        doc_id = get_doc_id(filename)

        articles = split_gk4_into_articles(text)
        if not articles:
            logging.warning(
                f"{filename}: GK4 detection failed, falling back to generic splitting"
            )
            parent_chunks = recursive_split(text, PARENT_CHUNK_SIZE, CHUNK_OVERLAP)
            total_parents = len(parent_chunks)
            total_children = 0
            for p_index, p_text in enumerate(parent_chunks):
                parent_id = save_parent_to_store(doc_id, p_index, p_text, filename)
                child_chunks = recursive_split(
                    p_text, CHILD_CHUNK_SIZE, CHUNK_OVERLAP
                )
                for c_index, c_text in enumerate(child_chunks):
                    child_id = f"{parent_id}_c{c_index}"
                    metadata = {
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "filename": filename,
                        "part_index": p_index,
                        "total_parts": total_parents,
                        "page_content": c_text,
                    }
                    clean_meta = sanitize_metadata(metadata)
                    embedding = model.encode(c_text).tolist()
                    collection.add(
                        ids=[child_id],
                        embeddings=[embedding],
                        metadatas=[clean_meta],
                        documents=[c_text],
                    )
                    total_children += 1
            logging.info(
                f"{filename}: finished GK4 fallback parents={total_parents}, children={total_children}"
            )
            gc.collect()
            continue

        total_parents = len(articles)
        logging.info(f"{filename}: GK4 mode, articles={total_parents}")
        total_children = 0
        for p_index, (article_number, article_text) in enumerate(articles):
            parent_id = save_parent_to_store(doc_id, p_index, article_text, filename)
            child_chunks = split_gk4_article_children(article_text)
            logging.info(
                f"{filename}: article {article_number} ({p_index + 1}/{total_parents}) children={len(child_chunks)}"
            )
            for c_index, c_text in enumerate(child_chunks):
                child_id = f"{parent_id}_c{c_index}"
                metadata = {
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "part_index": p_index,
                    "total_parts": total_parents,
                    "page_content": c_text,
                    "article_number": article_number,
                }
                clean_meta = sanitize_metadata(metadata)
                embedding = model.encode(c_text).tolist()
                collection.add(
                    ids=[child_id],
                    embeddings=[embedding],
                    metadatas=[clean_meta],
                    documents=[c_text],
                )
                total_children += 1
        logging.info(
            f"{filename}: finished GK4 articles={total_parents}, children={total_children}"
        )
        gc.collect()


def ingest_documents():
    logging.info("Starting ingestion")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        logging.error(f"{KNOWLEDGE_BASE_DIR} not found")
        return

    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing_files = set()
    try:
        existing_data = collection.get(include=["metadatas"])
        for meta in existing_data.get("metadatas", []):
            if meta and "filename" in meta:
                existing_files.add(meta["filename"])
    except Exception as e:
        logging.error(f"Error reading existing metadata: {e}")

    model = load_embedding_model()

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if is_gk4_filename(filename):
            logging.info(f"Skipping GK4 file in main collection: {filename}")
            continue

        if filename in existing_files:
            logging.info(f"Skipping already indexed file: {filename}")
            continue

        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        logging.info(f"Processing file: {filename}")

        text = None
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = read_pdf(filepath)
        elif lower.endswith(".docx"):
            text = read_docx(filepath)
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            text = read_excel(filepath)

        if not text:
            logging.error(f"No text extracted from {filename}, skipping")
            continue

        doc_id = get_doc_id(filename)

        parent_chunks = recursive_split(text, PARENT_CHUNK_SIZE, CHUNK_OVERLAP)
        total_parents = len(parent_chunks)
        logging.info(f"{filename}: parent chunks={total_parents}")

        total_children = 0

        for p_index, p_text in enumerate(parent_chunks):
            parent_id = save_parent_to_store(doc_id, p_index, p_text, filename)

            child_chunks = recursive_split(p_text, CHILD_CHUNK_SIZE, CHUNK_OVERLAP)
            logging.info(
                f"{filename}: parent {p_index + 1}/{total_parents} children={len(child_chunks)}"
            )

            for c_index, c_text in enumerate(child_chunks):
                child_id = f"{parent_id}_c{c_index}"
                metadata = {
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "part_index": p_index,
                    "total_parts": total_parents,
                    "page_content": c_text,
                    "prev_parent_id": f"{doc_id}_{p_index - 1}" if p_index > 0 else None,
                    "next_parent_id": f"{doc_id}_{p_index + 1}"
                    if p_index < total_parents - 1
                    else None,
                }
                clean_meta = sanitize_metadata(metadata)
                embedding = model.encode(c_text).tolist()
                collection.add(
                    ids=[child_id],
                    embeddings=[embedding],
                    metadatas=[clean_meta],
                    documents=[c_text],
                )
                total_children += 1

        logging.info(
            f"{filename}: finished parents={total_parents}, children={total_children}"
        )
        gc.collect()

    logging.info("Ingestion complete")


if __name__ == "__main__":
    ingest_documents()
