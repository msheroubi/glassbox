#!/usr/bin/env bash
# Real SO-101 follower arm + overhead camera, driven by a trained LeRobot ACT checkpoint.
#   scripts/start_robot.sh <policy path or HF repo id> [extra backend args...]
#   e.g. scripts/start_robot.sh models/last/pretrained_model --camera /dev/video2 --robot-id kowalski_follower \
#          --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082778-if00
#   Use the follower's /dev/serial/by-id path: /dev/ttyACM0/1 swap between leader and follower on re-plug.
# Defaults: overhead camera captured at 640x360 and no wrist camera. Override with
# --camera-size WxH and --wrist-camera INDEX_OR_PATH (later flags win).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [ $# -lt 1 ]; then
  echo "usage: $0 <policy path> [--robot-port /dev/ttyACM0] [--robot-id ID] [--camera INDEX_OR_PATH] ..." >&2
  exit 2
fi
POLICY="$1"
shift
echo "GlassBox-SO101 (SO-101 hardware, policy=$POLICY) -> http://localhost:${PORT:-8000}"
exec "$PYTHON" -m backend.main --policy-path "$POLICY" --port "${PORT:-8000}" --camera-size 640x360 --wrist-camera none "$@"
