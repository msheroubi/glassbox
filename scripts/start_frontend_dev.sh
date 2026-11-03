#!/usr/bin/env bash
# Vite dev server with hot reload on :5173, proxying /api and /ws to the backend.
#   GLASSBOX_BACKEND=http://localhost:8000 scripts/start_frontend_dev.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../frontend"
[ -d node_modules ] || npm install --no-fund --no-audit
exec npm run dev
