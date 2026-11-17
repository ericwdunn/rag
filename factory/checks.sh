#!/usr/bin/env bash
# The repo's checks. Exit 0 only when the branch is fit to review.
#
# The gate runs this script from the main checkout with the job's worktree as
# the working directory. Keep it fast, deterministic and non-interactive.
set -euo pipefail

# Guard ADDED files too: protected_paths only catches modified/deleted ones.
# Nothing in this repo deploys, emails or charges on merge today; the only way
# a branch could add a trigger is a GitHub Actions workflow under .github/.
added=$(git diff --name-only origin/main...HEAD -- .github/)
if [ -n "$added" ]; then
  echo "Branch touches .github/ (CI/deploy triggers); a human must make that change:" >&2
  echo "$added" >&2
  exit 1
fi

# Fresh worktree: build its own .venv from the lockfile, never re-resolve.
uv sync --frozen --quiet
uv run --frozen python test_rag.py
