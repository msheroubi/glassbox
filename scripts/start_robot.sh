#!/usr/bin/env bash
# Real SO-101 follower arm + overhead camera, driven by a trained LeRobot ACT checkpoint.
#   scripts/start_robot.sh <policy path or HF repo id> [extra backend args...]
#   e.g. scripts/start_robot.sh outputs/train/act_so101/checkpoints/last/pretrained_model --robot-port /dev/ttyACM0
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [ $# -lt 1 ]; then
  echo "usage: $0 <policy path> [--robot-port /dev/ttyACM0] [--robot-id ID] ..." >&2
  exit 2
fi
POLICY="$1"
shift
echo "GlassBox-SO101 (SO-101 hardware, policy=$POLICY) -> http://localhost:${PORT:-8000}"
exec "$PYTHON" -m backend.main --policy-path "$POLICY" --port "${PORT:-8000}" "$@"
