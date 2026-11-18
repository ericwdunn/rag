import os
import sys
from pathlib import Path

import numpy as np
import psycopg
import requests

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"
DATABASE_URL = os.environ.get("RAG_DATABASE_URL", "dbname=rag")


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    return np.array(r.json()["embeddings"])


def ask_llm(prompt):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    })
    return r.json()["message"]["content"]


def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def load_chunks(docs_dir="docs"):
    return [c for file in sorted(Path(docs_dir).glob("*.md")) for c in split_paragraphs(file.read_text(encoding="utf-8"))]


# pgvector accepts a vector as the text literal '[x,y,...]', so no extra adapter package is needed.
def to_pgvector(v):
    return "[" + ",".join(str(float(x)) for x in v) + "]"


# Embed every docs/*.md paragraph once and store it. Each file's rows are replaced inside one
# transaction, so edited or removed paragraphs never leave stale rows behind.
def ingest(docs_dir="docs"):
    with psycopg.connect(DATABASE_URL) as conn:
        for file in sorted(Path(docs_dir).glob("*.md")):
            chunks = split_paragraphs(file.read_text(encoding="utf-8"))
            vectors = embed(chunks)
            conn.execute("DELETE FROM chunks WHERE source = %s", (str(file),))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (source, chunk_index, content, embedding) VALUES (%s, %s, %s, %s::vector)",
                    [(str(file), i, c, to_pgvector(v)) for i, (c, v) in enumerate(zip(chunks, vectors))],
                )
            print(f"{file}: {len(chunks)} chunks")


# Top-k chunks by cosine distance (<=>), computed in Postgres.
def retrieve(question, k=3):
    q = to_pgvector(embed([question])[0])
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute("SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s", (q, k)).fetchall()
    return [r[0] for r in rows]


def build_prompt(question, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}"""


def main():
    if sys.argv[1:] == ["ingest"]:
        return ingest()
    while True:
        question = input("Ask: ")
        print("\n" + ask_llm(build_prompt(question, retrieve(question))) + "\n")


if __name__ == "__main__":
    main()
