#!/usr/bin/env bash
# Camera-only + a pretrained VLA's attention maps: SmolVLA watches the overhead webcam and the console shows
# where it looks. Observe only: no arm is connected and no action is ever sent.
#   scripts/start_vla.sh [camera index or /dev/videoN] [extra backend args...]
#   e.g. scripts/start_vla.sh 2 --vla-attention language --concepts "target=red cube,gripper=robot arm,goal=green hex tile"
# VLA=<path or HF repo id> picks the checkpoint (default lerobot/smolvla_base, ~1 GB download on first run).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CAMERA="${1:-2}"
shift || true
VLA="${VLA:-lerobot/smolvla_base}"
echo "GlassBox-SO101 (camera-only, webcam $CAMERA, $VLA attention, observe only) -> http://localhost:${PORT:-8000}"
exec "$PYTHON" -m backend.main --webcam "$CAMERA" --vla "$VLA" --port "${PORT:-8000}" "$@"
