# Smart Librarian – RAG + Tool (CLI)

Acest proiect implementează assignment-ul de LLM Integratons (ChromaDB cu embeddings OpenAI, retriever semantic, chatbot + tool tool usage)

## Arhitectură

- **Open Library API** pentru a genera un JSON cu 100+ cărți (`src/fetch_books_openlibrary.py`).
- **ChromaDB** (persistat local în `./chroma_db`) încărcat cu embeddings **`text-embedding-3-small`**.
- **Retriever semantic** după temă/context + **chatbot CLI** care integrează OpenAI GPT și tool-ul `get_summary_by_title`.
- **Format JSON**: `data/books.json` cu câmpuri cheie: `id`, `title`, `authors`, `subjects`, `year`, `description`, `themes`, `summary`.

## Rulare

1. **Dependențe**
   ```bash
   python -m venv .venv && .venv\Scripts\activate # Unix: source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Cheia OpenAI daca nu va fi adaugat in .env**
   ```bash
   export OPENAI_API_KEY=your_key_here  # Windows: set OPENAI_API_KEY=...
   ```
3. **(Opțional) configurații env**
   Creați un `.env` în rădăcina proiectului pentru override:
   ```env
   # Optional cheia pentru OpenAI
   OPENAI_MODEL_GPT=gpt-4o-mini
   OPENAI_MODEL_EMB=text-embedding-3-small
   CHROMA_PATH=./chroma_db
   TOP_K=5
   ```
4. **Generează 100+ cărți în JSON**
   ```bash
   python src/fetch_books_openlibrary.py --limit 150
   ```
   Rezultatul: `data/books.json`
5. **Ingest în ChromaDB cu embeddings**
   ```bash
   python src/ingest_chroma.py --input data/books.json
   ```
6. **Rulează chatbot-ul CLI (RAG + tool)**
   ```bash
   python src/app_cli.py
   ```
   Exemple întrebări: „Vreau o carte despre prietenie și magie”, „Ce recomanzi pentru cineva care iubește povești de război?”

## Note

- `get_summary_by_title` caută în JSON-ul local și returnează rezumatul complet.
- Pentru a modifica numarul de recomandari primite la fiecare query modificati variabila de environment TOP_K
