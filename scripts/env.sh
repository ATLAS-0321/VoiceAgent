#!/usr/bin/env bash
# Source this before any voiceagent command.
# Sets ROCm/HIP env vars for gfx1100 (RX 7900 XTX).

# RDNA3 architecture override (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export ROCM_HOME=/opt/rocm

# AMD perf tuning (from groxaxo/Chatterbox-TTS-Server analysis)
export HIPBLASLT_USE_HEURISTIC=1
export GPU_MAX_HW_QUEUES=1

# MIOpen kernel cache — avoids 12+ second cold-start recompiles per session.
# Persisted between runs; safe to delete to force re-tune.
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH="$HOME/.cache/miopen"
mkdir -p "$MIOPEN_USER_DB_PATH" 2>/dev/null || true

# Reduce log noise from libraries
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_TELEMETRY=1

# HuggingFace bleibt im System-Default (~/.cache/huggingface).
# Unsere Modelle leben project-lokal in models/qwen3-asr/ + models/qwen3-tts-base/
# (per snapshot_download local_dir= in install.sh). HF_HOME hier zu setzen
# wuerde HF-Tokens nach models/hf/ schreiben — wollen wir nicht.

# Repo root — dynamisch aus Script-Lokation, funktioniert egal wo der
# Repo gecloned wurde (auch ohne deutsche Locale-Annahme).
export VOICEAGENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
