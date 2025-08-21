import json, time, argparse, os
from collections import OrderedDict
from typing import List, Dict, Optional
import requests
from tqdm import tqdm

DEFAULT_TOPICS = [
    "fantasy", "science_fiction", "friendship", "love", "adventure", "war",
    "mystery", "magic", "history", "philosophy", "psychology", "dystopia",
    "coming_of_age", "thriller", "crime", "biography", "self_help", "horror"
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "smart-librarian/1.0 (edu demo)"})
REST_BASE = "https://openlibrary.org"

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x); out.append(x)
    return out

def _safe_desc_field(desc) -> Optional[str]:
    # Open Library can return description as string or {"value": "..."} or list
    if not desc:
        return None
    if isinstance(desc, str):
        return desc.strip() or None
    if isinstance(desc, dict) and "value" in desc:
        val = desc.get("value")
        return (val or "").strip() or None
    if isinstance(desc, list) and desc:
        # take first non-empty string-ish value
        for d in desc:
            if isinstance(d, str) and d.strip():
                return d.strip()
            if isinstance(d, dict) and "value" in d and d["value"]:
                return d["value"].strip()
    return None

def fetch_description_by_work_key(work_key: str) -> Optional[str]:
    # Example: /works/OL45883W.json
    url = f"{REST_BASE}{work_key}.json"
    r = SESSION.get(url, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    return _safe_desc_field(data.get("description"))

def fetch_description_fallback_from_edition(book_key: str) -> Optional[str]:
    # Example: /books/OL12345M.json -> may contain 'works': [{'key': '/works/OL...W'}]
    url = f"{REST_BASE}{book_key}.json"
    r = SESSION.get(url, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    # Try edition description directly
    ed_desc = _safe_desc_field(data.get("description"))
    if ed_desc:
        return ed_desc
    # Otherwise follow first work
    works = data.get("works") or []
    if isinstance(works, list) and works and isinstance(works[0], dict) and works[0].get("key"):
        return fetch_description_by_work_key(works[0]["key"])
    return None

def fetch_subject(topic: str, limit: int = 50) -> List[Dict]:
    """Use Search API to get candidates (fast), then enrich with REST API for descriptions."""
    url = "https://openlibrary.org/search.json"
    params = {
        "q": topic,
        "fields": "key,title,author_name,first_publish_year,subject,language,ia,edition_count",
        "limit": limit
    }
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = []
    for idx, doc in enumerate(data.get("docs", [])):
        key = doc.get("key")  # '/works/OL...W' or '/books/OL...M'
        if not key:
            continue
        title = doc.get("title") or ""
        authors = doc.get("author_name") or []
        year = doc.get("first_publish_year")
        subjects = doc.get("subject") or []
        themes = list({(t or '').lower().replace(' ', '_') for t in subjects[:10]} | {topic})

        # Enrich: get real description via REST API (work preferred)
        description = None
        try:
            if key.startswith("/works/"):
                description = fetch_description_by_work_key(key)
            elif key.startswith("/books/"):
                description = fetch_description_fallback_from_edition(key)
        except Exception:
            description = None

        summary = f"Teme: {', '.join([t for t in themes[:6] if t])}. O carte asociată cu '{topic}'. Autor(i): {', '.join(authors[:3])}."
        items.append({
            "id": key,
            "title": title,
            "authors": authors,
            "year": year,
            "subjects": subjects[:20],
            "themes": [t for t in themes[:10] if t],
            "description": description,  # may be None if not available
            "summary": summary
        })
        # polite throttle for REST hits
        time.sleep(0.05)
    return items

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150, help="Număr total minimal de cărți dorit")
    ap.add_argument("--per_topic", type=int, default=40, help="Câte rezultate/temă să cerem")
    ap.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS, help="Liste de subiecte/teme")
    ap.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "data", "books.json"))
    args = ap.parse_args()

    collected: Dict[str, dict] = OrderedDict()
    for topic in tqdm(args.topics, desc="Topics"):
        try:
            results = fetch_subject(topic, limit=args.per_topic)
        except Exception as e:
            print(f"[WARN] {topic}: {e}")
            continue
        for it in results:
            collected.setdefault(it["id"], it)
        time.sleep(0.2)

    books = list(collected.values())[: max(args.limit, 0) or len(collected)]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(books)} books to {args.output}")

if __name__ == "__main__":
    main()
