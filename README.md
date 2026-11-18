# rag

A small, fully local retrieval-augmented generation (RAG) pipeline, built to learn
how each piece works with no framework in the way. About 80 lines of Python, with Postgres + pgvector as the vector store.

## How it works

1. **Load:** reads every `docs/*.md` file and splits it into chunks on blank lines
   (one paragraph per chunk).
2. **Embed + store (once):** `uv run rag.py ingest` embeds the chunks with
   [Ollama](https://ollama.com) (`nomic-embed-text`, 768 dims) and stores text, source file
   and vector in a Postgres `chunks` table (see `schema.sql`). Re-running it replaces each
   file's rows in one transaction, so edited paragraphs never leave stale rows.
3. **Retrieve:** embeds your question and lets Postgres rank the chunks by cosine distance
   (pgvector's `<=>` operator), keeping the top 3. The table has an HNSW index; at a few dozen
   rows the planner rightly ignores it and scans, and it starts paying off at thousands of rows.
4. **Answer:** passes those chunks to `qwen3:4b-instruct` with instructions to answer
   **only** from that context, and to say "I don't know" when the answer isn't there.

Everything runs on your own machine. There are no API keys and no cloud calls.

## The test corpus

`docs/handbook.md` is a made-up employee handbook for a fictional coffee shop,
"Riverbend Coffee Company". The handbook has 12 single-topic paragraphs with specific facts
planted in them so retrieval is easy to check, for example:

- "What's the alarm code?" → 4827
- "When is the schedule posted?" → Thursdays by 5 PM
- "What's the 401(k) match?" → 50% up to 4%
- "How much PTO do I earn?" → 1 hour per 30 worked

`docs/menu.md` is the same shop's menu, and it is messier on purpose: a short intro, a markdown
price table (12 / 16 / 20 oz), bulleted food, milk and syrup lists, stray headings and a few
notes. The splitter keeps the whole table in one chunk and leaves headings as tiny chunks of
their own. It plants its own facts, for example:

- "How much is a 16 oz latte?" → $4.95
- "Does the store sell oat milk?" → yes, +$0.75
- "What is the seasonal drink?" → Maple Pecan Latte, October through January

A question neither doc can answer (e.g. "Who won the World Series?") should get
"I don't know."

After adding or changing anything in `docs/`, re-run `uv run rag.py ingest`; retrieval only
sees what is in the database.

## Run it

Requires [uv](https://docs.astral.sh/uv/), a running Ollama, and Postgres with
[pgvector](https://github.com/pgvector/pgvector) (e.g. `brew install postgresql@18 pgvector`).

```sh
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
createdb rag && psql -d rag -f schema.sql   # one time
uv run rag.py ingest                        # whenever docs/ changes
uv run rag.py
```

Connects to `dbname=rag` by default; set `RAG_DATABASE_URL` to point elsewhere.

Then type questions at the `Ask:` prompt. Ctrl-C to quit.

Offline checks (no Ollama needed):

```sh
uv run python test_rag.py
```

Eval (needs Ollama and an ingested database): retrieval hit rate in the top 3 and "I don't know" refusals.

```sh
uv run eval.py
```

## Known limitations

- Chunking is naive (one paragraph per chunk), and `k` is fixed at 3.
- `eval.py` scores retrieval hit rate and refusals on a small question set covering the handbook and the menu.

## Roadmap

- [x] Store embeddings in Postgres with pgvector, embed once, and index with HNSW
- [x] Eval script: hit rate of the right chunk in the top-k, plus refusal checks on questions the handbook can't answer
- [ ] Compare chunk sizes and `k` values against the eval
- [x] A second, messier document (`docs/menu.md`)
- [ ] Optional: Claude as the answer model
