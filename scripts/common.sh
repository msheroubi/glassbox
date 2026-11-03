# Shared by the start_*.sh scripts: repo root, interpreter, and a one-time frontend build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

if [ ! -f .env ] && [ -z "${MODEL_API_KEY:-}" ]; then
  echo "note: no .env / MODEL_API_KEY: every Meta model will use its local fallback (cp .env.example .env)"
fi

# The backend serves frontend/dist on its own port; build it once (or after `rm -rf frontend/dist`).
if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "building the frontend (first run)..."
    (cd frontend && { [ -d node_modules ] || npm install --no-fund --no-audit; } && npm run build)
  else
    echo "npm not found: the API runs, but the UI needs 'npm run build' in frontend/ (or scripts/start_frontend_dev.sh)"
  fi
fi
