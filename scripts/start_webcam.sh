#!/usr/bin/env bash
# Camera-only: a USB webcam as the overhead view, no arm connected (no joint/servo telemetry).
#   scripts/start_webcam.sh [camera index or /dev/videoN] [extra backend args...]
#   e.g. scripts/start_webcam.sh 2 --concepts "target=red cube,gripper=robot arm,coffee mug"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CAMERA="${1:-0}"
shift || true
echo "GlassBox-SO101 (camera-only, webcam $CAMERA) -> http://localhost:${PORT:-8000}"
exec "$PYTHON" -m backend.main --webcam "$CAMERA" --port "${PORT:-8000}" "$@"
