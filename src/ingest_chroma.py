import os, json, argparse
from typing import List, Dict
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()

OPENAI_MODEL_EMB = os.getenv("OPENAI_MODEL_EMB", "text-embedding-3-small")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = os.getenv("COLLECTION", "books")

def build_doc(record: Dict) -> str:
    # Compose a dense text field for embedding
    parts = [
        f"Title: {record.get('title','')}",
        "Authors: " + ", ".join(record.get("authors", [])[:5]),
        "Year: " + str(record.get("year", "")),
        "Themes: " + ", ".join(record.get("themes", [])[:10]),
        "Subjects: " + ", ".join(record.get("subjects", [])[:20]),
        "Summary: " + (record.get("summary") or ""),
        "Description: " + (record.get("description") or ""),
    ]
    return "\n".join([p for p in parts if p and p != "Description: "])

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    # Batch for efficiency
    resp = client.embeddings.create(model=OPENAI_MODEL_EMB, input=texts)
    return [d.embedding for d in resp.data]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path la books.json")
    ap.add_argument("--reset", action="store_true", help="Șterge colecția existentă înainte de reîncărcare")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records: List[Dict] = json.load(f)

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if args.reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare payloads
    ids, docs, metadatas = [], [], []

    def _to_meta_value(v):
        # Chroma metadata must be primitive or None
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join([str(x) for x in v if x is not None])
        return str(v)

    for r in records:
        rid = r.get("id") or r.get("title") or "unknown"
        ids.append(str(rid))

        docs.append(build_doc(r))

        # Ensure all metadata fields are never None and are strings or numbers
        year_val = r.get("year")
        if year_val is None:
            year_val = ""
        metadatas.append({
            "title": _to_meta_value(r.get("title")),
            "authors": _to_meta_value(r.get("authors", [])),
            "themes": _to_meta_value(r.get("themes", [])),
            "subjects": _to_meta_value(r.get("subjects", [])),
            "year": year_val,
        })

    # Compute embeddings with OpenAI and add to Chroma
    oa = OpenAI()
    # Chunk in groups to avoid large payloads
    B = 128
    for i in range(0, len(docs), B):
        batch_ids = ids[i:i+B]
        batch_docs = docs[i:i+B]
        batch_meta = metadatas[i:i+B]
        embeds = embed_texts(oa, batch_docs)
        collection.add(ids=batch_ids, embeddings=embeds, documents=batch_docs, metadatas=batch_meta)

    print(f"Ingested {len(ids)} documents into Chroma collection '{COLLECTION}' at {CHROMA_PATH}")

if __name__ == "__main__":
    main()
