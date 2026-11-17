import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    return np.array(r.json()["embeddings"])


def ask_llm(prompt):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    })
    return r.json()["message"]["content"]


def load_chunks(docs_dir="docs"):
    chunks = []
    for file in sorted(Path(docs_dir).glob("*.md")):
        chunks += [p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    return chunks


# Rank chunks by cosine similarity between the question vector and each chunk vector; return the top k.
def rank(q_vector, chunks, chunks_vectors, k=3):
    scores = chunks_vectors @ q_vector / (np.linalg.norm(chunks_vectors, axis=1) * np.linalg.norm(q_vector))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]


def retrieve(question, chunks, chunks_vectors, k=3):
    return rank(embed([question])[0], chunks, chunks_vectors, k)


def build_prompt(question, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}"""


def main():
    chunks = load_chunks()
    chunks_vectors = embed(chunks)
    print(f"Indexed {len(chunks)} chunks\n")
    while True:
        question = input("Ask: ")
        print("\n" + ask_llm(build_prompt(question, retrieve(question, chunks, chunks_vectors))) + "\n")


if __name__ == "__main__":
    main()
