# Retrieval hit rate + "I don't know" refusal eval on docs/handbook.md: `uv run eval.py` (needs Ollama). Exits 0.
import rag

# (question, string that occurs only in the paragraph that answers it)
RETRIEVAL_SET = [
    ("What are the store hours on weekdays?", "6:00 AM to 8:00 PM Monday through Friday"),
    ("What time does the opening shift start on weekdays?", "5:15 AM"),
    ("What's the alarm code?", "4827"),
    ("When is the schedule posted?", "every Thursday by 5:00 PM"),
    ("What happens if I miss a shift without calling in?", "no-call, no-show"),
    ("Do I get a meal break?", "unpaid 30-minute meal break"),
    ("What's the 401(k) match?", "matches 50 percent"),
    ("What should I wear to work?", "green Riverbend apron"),
    ("How cold does milk need to be kept?", "41 degrees Fahrenheit"),
    ("When is payday?", "every other Friday"),
]

REFUSAL_SET = ["Who won the World Series?", "What is the Wi-Fi password?",
               "Does the store sell oat milk?", "What is the capital of France?"]
# Models often reply with a curly apostrophe; normalize it before matching.
refused = lambda answer: "i don't know" in answer.lower().replace("\u2019", "'")  # noqa: E731


def main():
    chunks = rag.load_chunks()
    chunks_vectors = rag.embed(chunks)
    hits = passes = 0
    for question, expected in RETRIEVAL_SET:
        hits += (hit := any(expected in c for c in rag.retrieve(question, chunks, chunks_vectors, k=3)))
        print(f"{'hit ' if hit else 'miss'}  {question}")
    for question in REFUSAL_SET:
        answer = rag.ask_llm(rag.build_prompt(question, rag.retrieve(question, chunks, chunks_vectors, k=3)))
        passes += (ok := refused(answer))
        print(f"{'pass' if ok else 'fail'}  {question}")
    print(f"Retrieval hit rate: {hits}/{len(RETRIEVAL_SET)} ({round(100 * hits / len(RETRIEVAL_SET))}%)")
    print(f"Refusals: {passes}/{len(REFUSAL_SET)}")


if __name__ == "__main__":
    main()
