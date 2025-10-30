from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError

import os
import json
import time
import numpy as np
from openai import OpenAI
import faiss
from dotenv import load_dotenv
load_dotenv()

# ------------------ Config ------------------
INDEX_NAME = "FAQ"
STORE_DIR = Path("BE") / "AllVectorDB" /INDEX_NAME
INDEX_PATH = STORE_DIR / f"{INDEX_NAME}.index"  # FAISS index
META_PATH = STORE_DIR / "meta.json"             # id -> {question, answer}
INFO_PATH = STORE_DIR / "index_info.json"       # info indice (modello, dim, ts, count)

EMBED_MODEL = "text-embedding-3-small"  # 1536-dim
BATCH_SIZE = 128

# SCHEMA
class FAQIngestion(BaseModel):
    question: str
    answer: str
    id: int
    source: str | None = None  # Per ora fissato al nome del documento. Ho solo generato alcune domande e risposte


def read_file(path_json: Path) -> List[Dict[str, Any]]:
    with open(path_json, "r", encoding="utf8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Il file JSON deve contenere una lista di oggetti FAQ.")
    return data

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def embed_batches(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
    return np.array(vecs, dtype=np.float32)

def build_faiss_index(vectors: np.ndarray, ids: List[int], out_path: Path):
    
    if vectors.ndim != 2:
        raise ValueError("vectors deve essere una matrice (N, D).")
    if len(ids) != vectors.shape[0]:
        raise ValueError("ids e vectors devono avere stessa lunghezza.")

    # normalizza per cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    v_unit = vectors / norms

    dim = v_unit.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(v_unit, np.asarray(ids, dtype=np.int64))
    faiss.write_index(index, str(out_path))


def create_index(path_json: Path, output_db_path: Path):

    ensure_dir(output_db_path)

    # 1) Carica e valida
    raw = read_file(path_json)
    items: List[FAQIngestion] = []
    for i, it in enumerate(raw, start=1):
        try:
            items.append(FAQIngestion(**it))
        except ValidationError as e:
            raise ValueError(f"Item {i} invalido: {e}")

    if not items:
        raise ValueError("Nessuna FAQ valida trovata nel JSON.")

    # 2) Prepara SOLO le question per embedding
    questions = [x.question.strip() for x in items]
    ids = [int(x.id) for x in items]

    # 3) Metadati (question + answer) per id
    meta = {str(x.id): {"question": x.question.strip(), "answer": x.answer.strip()} for x in items}

    # 4) OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY non impostata.")
    client = OpenAI(api_key=api_key)

    # 5) Embedding delle sole question
    vectors = embed_batches(client, questions, EMBED_MODEL)  # (N, D)

    # 6) Persistenza laterale (question, answer, ids, vettori)
    # np.save(output_db_path / "embeddings.npy", vectors)  # opzionale, utile per debug
    with open(output_db_path / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(output_db_path / "items.jsonl", "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps({"id": x.id, "question": x.question, "answer": x.answer}, ensure_ascii=False) + "\n")

    # 7) FAISS (id, vector)
    build_faiss_index(vectors, ids, INDEX_PATH)

    # 8) Info
    info = {
        "index_name": INDEX_NAME,
        "model": EMBED_MODEL,
        "count": len(ids),
        "dim": int(vectors.shape[1]),
        "created_at": int(time.time()),
        "source_file": str(path_json),
        "faiss_index_path": str(INDEX_PATH),
        "similarity": "cosine (normalized vectors with inner product)",
        "fields": ["id", "question", "answer", "vector"]  # schema logico richiesto
    }
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # print(f"[OK] Indice FAISS creato (SOLO question) in: {output_db_path.resolve()}")
    # print(f" - {INDEX_PATH}")
    # print(" - meta.json (id -> question, answer)")
    # print(" - items.jsonl")
    # print(" - embeddings.npy (debug)")
    # print(" - index_info.json")

# ------------------ Esempio d'uso diretto ------------------
if __name__ == "__main__":
    default_in = Path("BE") / "Ingestion" / "Documents" /"FAQs" / "FAQ.json"
    default_out = STORE_DIR  # BE/FAQ
    create_index(default_in, default_out)
