# Retrieval hit rate + "I don't know" refusal eval on every docs/*.md (handbook and menu): `uv run eval.py` (needs Ollama). Exits 0.
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
    ("How much is a 16 oz latte?", "| Latte | $4.15 | $4.95 |"),
    ("Does the store sell oat milk?", "Oat milk (+$0.75)"),
    ("What is the seasonal drink?", "Maple Pecan Latte"),
    ("Do you have anything gluten-free?", "Flourless chocolate cake (gluten-free)"),
    ("Is there a sugar-free syrup?", "sugar-free vanilla"),
    ("Is the food safe for someone with a nut allergy?", "shared kitchen that handles tree nuts"),
    ("How much does an extra espresso shot cost?", "Extra shot +$0.90"),
]

REFUSAL_SET = ["Who won the World Series?", "What is the Wi-Fi password?",
               "Does the store have a drive-through?", "What is the capital of France?"]
# Models often reply with a curly apostrophe; normalize it before matching.
refused = lambda answer: "i don't know" in answer.lower().replace("\u2019", "'")  # noqa: E731


def main():
    hits = passes = 0
    for question, expected in RETRIEVAL_SET:
        hits += (hit := any(expected in c for c in rag.retrieve(question, k=3)))
        print(f"{'hit ' if hit else 'miss'}  {question}")
    for question in REFUSAL_SET:
        answer = rag.ask_llm(rag.build_prompt(question, rag.retrieve(question, k=3)))
        passes += (ok := refused(answer))
        print(f"{'pass' if ok else 'fail'}  {question}")
    print(f"Retrieval hit rate: {hits}/{len(RETRIEVAL_SET)} ({round(100 * hits / len(RETRIEVAL_SET))}%)")
    print(f"Refusals: {passes}/{len(REFUSAL_SET)}")


if __name__ == "__main__":
    main()
