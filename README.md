# rag

A small, fully local retrieval-augmented generation (RAG) pipeline, built to learn
how each piece works with no framework in the way. About 40 lines of Python.

## How it works

1. **Load:** reads every `docs/*.md` file and splits it into chunks on blank lines
   (one paragraph per chunk).
2. **Embed:** sends the chunks to [Ollama](https://ollama.com) using the
   `nomic-embed-text` embedding model.
3. **Retrieve:** embeds your question and ranks the chunks by cosine similarity
   (numpy), keeping the top 3.
4. **Answer:** passes those chunks to `qwen3:4b-instruct` with instructions to answer
   **only** from that context, and to say "I don't know" when the answer isn't there.

Everything runs on your own machine. There are no API keys and no cloud calls.

## The test corpus

`docs/handbook.md` is a made-up employee handbook for a fictional coffee shop,
"Riverbend Coffee Company". It has 12 single-topic paragraphs with specific facts planted
in them so retrieval is easy to check, for example:

- "What's the alarm code?" → 4827
- "When is the schedule posted?" → Thursdays by 5 PM
- "What's the 401(k) match?" → 50% up to 4%
- "How much PTO do I earn?" → 1 hour per 30 worked

A question the handbook can't answer (e.g. "Who won the World Series?") should get
"I don't know."

## Run it

Requires [uv](https://docs.astral.sh/uv/) and a running Ollama.

```sh
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
uv run rag.py
```

Then type questions at the `Ask:` prompt. Ctrl-C to quit.

## Known limitations

- Embeddings live in memory and get rebuilt from scratch on every start.
- Chunking is naive (one paragraph per chunk), and `k` is fixed at 3.
- No evaluation yet, so retrieval quality is judged by eye.

## Roadmap

- [ ] Store embeddings in Postgres with pgvector, embed once, and index with HNSW
- [ ] Eval script: hit rate of the right chunk in the top-k, plus refusal checks on questions the handbook can't answer
- [ ] Compare chunk sizes and `k` values against the eval
- [ ] Optional: Claude as the answer model; a second, messier document
