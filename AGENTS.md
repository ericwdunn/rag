# rag: agent notes

A small local RAG build. `rag.py` loads `docs/*.md`, splits on blank lines, embeds with Ollama
`nomic-embed-text`, ranks chunks by cosine similarity (top k=3), and answers with `qwen3:4b-instruct`,
saying "I don't know" when the context doesn't cover the question. `docs/handbook.md` and
`docs/menu.md` (a messier price table, lists and notes) hold planted facts (e.g. alarm code 4827, the
16 oz latte price) that the tests and eval depend on.

## Setup

- Fresh checkout: `uv sync --frozen`. No env files.
- Run: `uv run rag.py` (needs Ollama running locally with both models pulled). No dev server or port.

## Validation

- `bash factory/checks.sh`, which runs `uv sync --frozen` and `uv run --frozen python test_rag.py`
  (offline, no Ollama; prints `ok`, exits non-zero on failure).
- `uv run eval.py` runs the retrieval hit-rate and refusal eval. It needs Ollama and is not part of
  `factory/checks.sh`.
- Tests run as a plain script, not via pytest discovery. If pytest is ever added, set
  `testpaths` so it never collects `.worktrees/`.

## What a push to `main` triggers

Nothing. There are no GitHub Actions, deploy scripts, email or billing hooks. Adding anything under
`.github/` changes that, so `factory/checks.sh` fails any factory branch that touches `.github/`.

## Software factory

`/factory run` builds every open GitHub issue labeled `factory` through
intake → spec → build → gate → review → approve, one PR per issue. State is in
`factory/jobs/` (gitignored); `factory/bin/job` is its only writer. Checks:
`factory/checks.sh`. Dashboard: `cd factory && python3 -m http.server`, then
open http://localhost:8000/dashboard.html.

Each job works in its own git worktree, a full checkout of the repo at
`.worktrees/<job-id>` (gitignored). A tool run from the repo root that doesn't
read `.gitignore` sees those copies too; exclude `.worktrees/` in its config.
