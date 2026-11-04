#!/usr/bin/env bash
# The hackathon demo: the real SO-101 runs ACT live; when it fails, GlassBox replays the recorded run offline
# (SAM 3.1 masks, plus an observe-only lens: SAM 3 locally by default), and Muse Wonder explains (and, if no
# human is needed, fixes) it.
#   scripts/start_demo.sh [extra backend args...]
# Environment (defaults are this rig's):
#   POLICY=../models/last/pretrained_model   ACT checkpoint
#   ROBOT_PORT=/dev/serial/by-id/...5A46082778-if00   follower arm (ttyACM numbers swap on re-plug)
#   ROBOT_ID=kowalski_follower   CAMERA=2 (overhead webcam)
#   LENS=sam3   replay lens: sam3 = facebook/sam3 run locally in .venv-sam3 (where it finds "red cube" and
#               "green hex tile"; perception, not attention), smolvla = $VLA's attention, none = ACT's only
#   VLA=lerobot/smolvla_base   the SmolVLA checkpoint for LENS=smolvla
#   STAGING=1   show the policy the red book on the goal wherever it is (labelled DEMO STAGING in the UI);
#               STAGING=0 when you place the book on the green tile yourself
#   AUTOSTART=0   wait for "▶ Start policy" in the console (after launch and each Reset); AUTOSTART=1 runs at once
#   FORCE_FAILURE=1   pressing Analyze (or saying "analyze") counts as a failure even if nothing was detected
#   SELF_CORRECT=1   apply Wonder's fix automatically; SELF_CORRECT=0 shows an "Apply" button instead
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
POLICY="${POLICY:-$ROOT/../models/last/pretrained_model}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-kowalski_follower}"
CAMERA="${CAMERA:-2}"
VLA="${VLA:-lerobot/smolvla_base}"
LENS="${LENS:-none}"

if [ ! -e "$ROBOT_PORT" ]; then
  echo "follower arm not found at $ROBOT_PORT (ls /dev/serial/by-id/; set ROBOT_PORT=...)" >&2
  exit 1
fi
if [ ! -d "$POLICY" ]; then
  echo "note: $POLICY is not a local directory; treating it as a Hugging Face repo id"
fi

ARGS=(--policy-path "$POLICY" --robot-port "$ROBOT_PORT" --robot-id "$ROBOT_ID" --camera "$CAMERA"
  --camera-size 640x360 --wrist-camera none --port "${PORT:-8000}")
case "$LENS" in
  sam3)
    if [ ! -x "$ROOT/.venv-sam3/bin/python" ]; then
      echo "LENS=sam3 needs $ROOT/.venv-sam3 (docs/06-running.md); or run with LENS=smolvla" >&2
      exit 1
    fi
    ARGS+=(--replay-sam3); LENS_NAME="SAM 3" ;;
  smolvla) ARGS+=(--replay-vla "$VLA"); LENS_NAME="$VLA" ;;
  none) LENS_NAME="no" ;;
  *) echo "LENS must be sam3, smolvla or none, not $LENS" >&2; exit 1 ;;
esac
[ "${STAGING:-1}" = 1 ] && ARGS+=(--demo-distractor-on-goal)
[ "${AUTOSTART:-0}" = 1 ] || ARGS+=(--start-paused)
[ "${FORCE_FAILURE:-1}" = 1 ] && ARGS+=(--demo-manual-failure)
[ "${SELF_CORRECT:-1}" = 1 ] && ARGS+=(--self-correct)

echo "GlassBox-SO101 demo (ACT live, $LENS_NAME replay lens, autostart=${AUTOSTART:-0}, staging=${STAGING:-1}, force-failure=${FORCE_FAILURE:-1}, self-correct=${SELF_CORRECT:-1})"
echo "  -> http://localhost:${PORT:-8000}   (saved failures: http://localhost:${PORT:-8000}/#replay)"
exec "$PYTHON" -m backend.main "${ARGS[@]}" "$@"
