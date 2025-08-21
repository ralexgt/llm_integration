import os, json, argparse, sys
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import chromadb
from openai import OpenAI

load_dotenv()
console = Console()

OPENAI_MODEL_GPT = os.getenv("OPENAI_MODEL_GPT", "gpt-4o-mini")
OPENAI_MODEL_EMB = os.getenv("OPENAI_MODEL_EMB", "text-embedding-3-small")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = os.getenv("COLLECTION", "books")
TOP_K = int(os.getenv("TOP_K", "5"))
BOOKS_JSON = os.getenv("BOOKS_JSON", os.path.join(os.path.dirname(__file__), "..", "data", "books.json"))

# ---- Pricing maps (USD) ----
# Actualizează dacă schimbi modelele; valorile sunt estimative.
PRICE_PER_M_IN = {
    "gpt-4o": 5.00,
    "gpt-4o-mini": 0.60,
}
PRICE_PER_M_OUT = {
    "gpt-4o": 20.00,
    "gpt-4o-mini": 2.40,
}
# Embeddings (pret / 1K tokens)
EMB_PRICE_PER_1K = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
}

def _cost_chat(model: str, in_tok: int, out_tok: int) -> float:
    c_in = PRICE_PER_M_IN.get(model)
    c_out = PRICE_PER_M_OUT.get(model)
    if c_in is None or c_out is None:
        return 0.0
    return (in_tok / 1_000_000) * c_in + (out_tok / 1_000_000) * c_out

def _cost_embed(model: str, tok: int) -> float:
    per_1k = EMB_PRICE_PER_1K.get(model)
    if per_1k is None:
        return 0.0
    return (tok / 1_000) * per_1k

def get_summary_by_title(title: str) -> str:
    # Tool local: caută în JSON rezumatul complet (sau fallback la summary)
    try:
        with open(BOOKS_JSON, "r", encoding="utf-8") as f:
            books = json.load(f)
    except Exception:
        return "Rezumat indisponibil momentan."
    for b in books:
        if b.get("title","").strip().lower() == title.strip().lower():
            return b.get("description") or b.get("summary") or "Rezumat indisponibil."
    return "Rezumat indisponibil."

def search_similar(query: str):
    # Embedează întrebarea și interoghează Chroma pentru TOP_K
    oa = OpenAI()
    emb_resp = oa.embeddings.create(model=OPENAI_MODEL_EMB, input=query)
    emb = emb_resp.data[0].embedding

    # usage embeddings (dacă SDK îl oferă)
    emb_tokens = 0
    try:
        emb_tokens = getattr(getattr(emb_resp, "usage", None), "total_tokens", 0) or \
                     getattr(getattr(emb_resp, "usage", None), "prompt_tokens", 0) or 0
    except Exception:
        emb_tokens = 0

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    res = col.query(
        query_embeddings=[emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    items = []
    for i in range(len(res["ids"][0])):
        items.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return items, emb_tokens

def build_prompt(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # Construiește context minimal (doar metadate utile)
    context_blocks = []
    for c in candidates[:TOP_K]:
        md = c["metadata"]
        context_blocks.append(
            f"- Titlu: {md.get('title')} | Autori: {', '.join(str(md.get('authors', '')).split(',')[:3]).strip()} "
            f"| Teme: {', '.join(str(md.get('themes', '')).split(',')[:5]).strip()}"
        )
    context = "\n".join(context_blocks)

    # Prompt strict: doar în română, un singur titlu, format fix + JSON de titlu la final
    sys_prompt = (
        "Ești un bibliotecar AI care răspunde STRICT în limba română. "
        "Primești o întrebare a utilizatorului despre interese/teme și o listă de cărți candidate dintr-un vector store. "
        "Trebuie să recomanzi O SINGURĂ carte. "
        "Formatul răspunsului trebuie să fie exact acesta:\n\n"
        "- recomandare\n"
        "{titlu}\n"
        "{rezumat}\n\n"
        "La final, include și un mic JSON pe o linie separată, de forma {\"title\": \"Titlu Exact\"}, "
        "pentru a putea apela tool-ul de rezumat."
    )
    user_prompt = (
        f"Întrebare: {query}\n\n"
        f"Cărți candidate (rezumate în metadate):\n{context}\n\n"
        "Respectă exact formatul cerut mai sus și nu adăuga alte elemente."
    )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

def chat_once(query: str):
    oa = OpenAI()
    candidates, emb_tokens = search_similar(query)
    messages = build_prompt(query, candidates)
    resp = oa.chat.completions.create(
        model=OPENAI_MODEL_GPT,
        messages=messages,
        temperature=0.6,
    )
    text = resp.choices[0].message.content.strip()

    # usage chat
    in_tok = getattr(getattr(resp, "usage", None), "prompt_tokens", 0)
    out_tok = getattr(getattr(resp, "usage", None), "completion_tokens", 0)
    total_tok = getattr(getattr(resp, "usage", None), "total_tokens", in_tok + out_tok)

    # costuri
    chat_cost = _cost_chat(OPENAI_MODEL_GPT, in_tok, out_tok)
    emb_cost = _cost_embed(OPENAI_MODEL_EMB, emb_tokens) if emb_tokens else 0.0

    # Extrage titlul din JSON-ul de pe ultima linie
    chosen_title = None
    for line in text.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}") and "title" in line.lower():
            try:
                chosen_title = json.loads(line).get("title")
            except Exception:
                pass
            break
    if not chosen_title and candidates:
        chosen_title = candidates[0]["metadata"].get("title")

    usage = (in_tok, out_tok, total_tok, emb_tokens, chat_cost, emb_cost)
    return text, usage, (chosen_title or "")

def main():
    console.print("[bold green]Smart Librarian CLI[/bold green] — scrie întrebarea ta (Ctrl+C pentru ieșire)")
    while True:
        try:
            q = console.input("[bold cyan]?[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nLa revedere!")
            break
        if not q.strip():
            continue

        answer, usage, title = chat_once(q)
        # Afișează răspunsul modelului (deja în formatul cerut)
        console.print(Markdown(answer))

        # Afișează rezumatul complet pentru titlul ales
        if title:
            summary = get_summary_by_title(title)
            console.print(Markdown(f"\n**Rezumat pentru:** _{title}_\n{summary}"))

        # Afișează usage + costuri
        in_tok, out_tok, total_tok, emb_tok, chat_cost, emb_cost = usage
        total_cost = chat_cost + emb_cost
        console.print(
            f"\n[dim]Tokeni chat: in={in_tok}, out={out_tok}, total={total_tok} | "
            f"Tokeni embeddings: {emb_tok} | "
            f"Cost estimat: chat ${chat_cost:.6f} + emb ${emb_cost:.6f} = "
            f"[bold]${total_cost:.6f}[/bold] "
            f"(modele: {OPENAI_MODEL_GPT}, {OPENAI_MODEL_EMB})[/dim]"
        )

if __name__ == "__main__":
    main()
