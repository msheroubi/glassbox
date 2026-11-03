#!/usr/bin/env bash
# Simulated SO-101 + overhead camera, no hardware.
#   scripts/start_mock.sh [scenario] [extra backend args...]
#   scenarios: nominal | attention_hijack | servo_stall | occlusion | specular_glare
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCENARIO="${1:-attention_hijack}"
shift || true
echo "GlassBox-SO101 (mock, scenario=$SCENARIO) -> http://localhost:${PORT:-8000}"
exec "$PYTHON" -m backend.main --mock-hardware --scenario "$SCENARIO" --port "${PORT:-8000}" "$@"
