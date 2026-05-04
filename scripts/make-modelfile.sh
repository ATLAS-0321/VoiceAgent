#!/usr/bin/env bash
# Create the custom Ollama model from its Modelfile.
#   gemma4-data : Gemma 4 with thinking disabled (raw=true via /api/generate
#                 in voiceagent/llm.py — the Modelfile only sets temperature
#                 + parameters, no system-prompt; personas are file-based.)
# Idempotent — safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama not installed"
    exit 1
fi

create_model () {
    local name="$1"
    local modelfile="$2"

    if [ ! -f "$modelfile" ]; then
        echo "ERROR: Modelfile not found: $modelfile"
        return 1
    fi

    echo ""
    echo "Creating Ollama model '$name' from $(basename "$modelfile")..."
    ollama create "$name" -f "$modelfile"

    local response
    response="$(ollama run "$name" --think=false 'Sag in einem Satz Hallo' 2>/dev/null)"
    echo "  Response: $response"
    if echo "$response" | grep -qiE "thinking|analyze|step 1|process:"; then
        echo "WARNING: Response still contains thinking-style keywords."
        return 2
    fi
    echo "  OK - no thinking output detected."
}

create_model "gemma4-data" "$ROOT/scripts/Modelfile.gemma4-data"

echo ""
echo "Done. Active model in config.yaml: gemma4-data."
