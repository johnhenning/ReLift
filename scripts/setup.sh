#!/usr/bin/env bash
# One-shot setup: pinned Python dependencies and an editable project install with uv.
#
#     bash scripts/setup.sh
#     uv run python scripts/download_weights.py
#
# Rerunning synchronizes the environment with uv.lock. Set UV_PROJECT_ENVIRONMENT
# for a different environment directory, or UV_CACHE_DIR to relocate uv's cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
if (( $# )); then
    echo "Usage: bash scripts/setup.sh (set UV_PROJECT_ENVIRONMENT for a custom directory)" >&2
    exit 1
fi

echo "== uv environment"
uv sync --locked --extra dev

echo "== check"
uv run --no-sync python -c "import relift, torch; print('relift ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "done. Next: uv run python scripts/download_weights.py"
