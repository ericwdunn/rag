# Offline checks (no Ollama needed): run with `uv run python test_rag.py`. Exits non-zero on failure.
import numpy as np
import rag

chunks = rag.load_chunks()
assert len(chunks) == 12, f"expected 12 handbook paragraphs, got {len(chunks)}"
assert any("4827" in c for c in chunks), "planted alarm code missing from handbook"

# Cosine ranking: the chunk pointing the same way as the question wins, regardless of magnitude.
vecs = np.array([[1.0, 0.0], [0.0, 5.0], [0.7, 0.7]])
assert rag.rank(np.array([0.0, 1.0]), ["a", "b", "c"], vecs, k=2) == ["b", "c"]

prompt = rag.build_prompt("What is the alarm code?", ["ctx one", "ctx two"])
assert "ctx one\n\nctx two" in prompt and "I don't know" in prompt and prompt.endswith("What is the alarm code?")

# Eval sets: each expected string must pin down exactly one chunk, or a "hit" means nothing.
import eval
for _, expected in eval.RETRIEVAL_SET:
    assert sum(expected in c for c in chunks) == 1, f"{expected!r} must occur in exactly one chunk"
assert 10 <= len(eval.RETRIEVAL_SET) <= 15 and 3 <= len(eval.REFUSAL_SET) <= 5, "eval set sizes out of range"

print("ok")
