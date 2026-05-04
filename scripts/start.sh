#!/usr/bin/env bash
# Start the realtime voice loop.
#   ./scripts/start.sh                       # use config.yaml
#   ./scripts/start.sh --voice <name>        # override tts.voice (subfolder in voices/)
#   ./scripts/start.sh --list-devices        # show audio devices
#   ./scripts/start.sh --list-voices         # show available voices
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

if [ ! -d "$ROOT/.venv" ]; then
    echo "ERROR: venv missing. Run ./install.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

cd "$ROOT"
exec python -m voiceagent "$@"
