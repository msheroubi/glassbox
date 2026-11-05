#!/usr/bin/env bash
# The GlassBox-SO101 demo: the real SO-101 runs ACT live; when it fails, GlassBox replays the recorded run offline
# (SAM 3.1 masks, plus SAM 3 run locally as a perception lens), and Muse Wonder explains (and, if no human is
# needed, fixes) it. Builds the console on first run, then serves it on http://localhost:$PORT.
#   scripts/start_demo.sh [extra backend args...]
# Environment (defaults are this rig's):
#   PYTHON=python   the lerobot environment's interpreter
#   POLICY=../models/last2/pretrained_model   ACT checkpoint (local dir or HF repo id)
#   ROBOT_PORT=/dev/serial/by-id/...5A46082778-if00   follower arm (ttyACM numbers swap on re-plug)
#   ROBOT_ID=kowalski_follower   CAMERA=2 (overhead webcam)   PORT=8000
#   STAGING=1   show the policy the red book on the goal wherever it is (labelled DEMO STAGING in the UI);
#               STAGING=0 when you place the book on the green tile yourself
#   AUTOSTART=0   wait for "▶ Start policy" in the console (after launch and each Reset); AUTOSTART=1 runs at once
#   FORCE_FAILURE=1   pressing Analyze (or saying "analyze") counts as a failure even if nothing was detected
#   SELF_CORRECT=1   apply Wonder's fix automatically; SELF_CORRECT=0 shows an "Apply" button instead
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

POLICY="${POLICY:-$ROOT/../models/last2/pretrained_model}"
ROBOT_PORT="${ROBOT_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082778-if00}"
ROBOT_ID="${ROBOT_ID:-kowalski_follower}"
CAMERA="${CAMERA:-2}"
PORT="${PORT:-8000}"

if [ ! -e "$ROBOT_PORT" ]; then
  echo "follower arm not found at $ROBOT_PORT (ls /dev/serial/by-id/; set ROBOT_PORT=...)" >&2
  exit 1
fi
if [ ! -d "$POLICY" ]; then
  echo "note: $POLICY is not a local directory; treating it as a Hugging Face repo id"
fi
if [ ! -x "$ROOT/.venv-sam3/bin/python" ]; then
  echo "the SAM 3 replay lens needs $ROOT/.venv-sam3 (see the README)" >&2
  exit 1
fi
if [ ! -f .env ] && [ -z "${MODEL_API_KEY:-}" ]; then
  echo "note: no .env / MODEL_API_KEY: every Meta model will use its local fallback (cp .env.example .env)"
fi

# The backend serves frontend/dist on its own port; build it once (or after `rm -rf frontend/dist`).
if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "building the frontend (first run)..."
    (cd frontend && { [ -d node_modules ] || npm install --no-fund --no-audit; } && npm run build)
  else
    echo "npm not found: the API runs, but the console needs 'npm run build' in frontend/"
  fi
fi

ARGS=(--policy-path "$POLICY" --robot-port "$ROBOT_PORT" --robot-id "$ROBOT_ID" --camera "$CAMERA"
  --camera-size 640x360 --wrist-camera none --port "$PORT")
[ "${STAGING:-1}" = 1 ] && ARGS+=(--demo-distractor-on-goal)
[ "${AUTOSTART:-0}" = 1 ] || ARGS+=(--start-paused)
[ "${FORCE_FAILURE:-1}" = 1 ] && ARGS+=(--demo-manual-failure)
[ "${SELF_CORRECT:-1}" = 1 ] && ARGS+=(--self-correct)

echo "GlassBox-SO101 demo (ACT live, SAM 3 replay lens, autostart=${AUTOSTART:-0}, staging=${STAGING:-1}, force-failure=${FORCE_FAILURE:-1}, self-correct=${SELF_CORRECT:-1})"
echo "  -> http://localhost:$PORT   (saved failures: http://localhost:$PORT/#replay)"
exec "$PYTHON" -m backend.main "${ARGS[@]}" "$@"
