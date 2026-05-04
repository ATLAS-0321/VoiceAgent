# Scripts

Shell and Python helpers for setup, start, build and tooling. All scripts
are executable (`chmod +x`) and source `env.sh` for ROCm/CUDA env vars.

## Overview

| File | Purpose |
|---|---|
| `start.sh` | Start the loop (loads env + venv, runs `python -m voiceagent`) |
| `env.sh` | ROCm/HIP env vars (gfx1100, MIOPEN cache) — sourced by all other scripts |
| `make-modelfile.sh` | Create the custom Ollama model `gemma4-data` from `Modelfile.gemma4-data` |
| `Modelfile.gemma4-data` | Ollama Modelfile (sampling params, Gemma 4 prompt format, no-thinking) — input for `make-modelfile.sh` |
| `verify-rocm.sh` | Standalone pre-flight check: rocminfo + PyTorch import + GPU matmul smoke test |
| `clean-reference.py` | Dirty source → clean voice sample (Demucs + Silero-VAD + Resemblyzer/KMeans) |

## start.sh

Default loop start:

```bash
./scripts/start.sh                       # uses config.yaml
./scripts/start.sh --voice Anna          # override tts.voice (subfolder in voices/)
./scripts/start.sh --list-devices        # list PortAudio devices
./scripts/start.sh --list-voices         # show available voices
```

Sources `env.sh`, activates `.venv`, runs `python -m voiceagent`.

## env.sh

ROCm/HIP tuning vars for gfx1100 (RX 7900 XTX). MUST be sourced before
PyTorch-ROCm initializes. Important:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` — RDNA3 architecture override
- `MIOPEN_USER_DB_PATH` + `MIOPEN_FIND_MODE=FAST` — persistent kernel
  cache eliminates ~12 s cold-start after reboot

Automatically sourced by `start.sh`, `verify-rocm.sh` and `install.sh` —
you typically don't touch it directly.

## make-modelfile.sh + Modelfile.gemma4-data

Creates the custom Ollama model `gemma4-data` from the Modelfile.
`install.sh` calls this automatically when the model doesn't exist yet.

Run manually if you changed the sampling params in the Modelfile:

```bash
./scripts/make-modelfile.sh
```

The Modelfile only sets sampling params + Gemma 4 prompt format —
NO system prompt. Personas are file-based under `voices/<voice>/persona.txt`.

## verify-rocm.sh

Pre-flight check when the GPU misbehaves — shows:
- ROCm version + GPU name
- PyTorch version + HIP variant
- 2048×2048 matmul smoke test (performance indicator)

```bash
./scripts/verify-rocm.sh
```

## clean-reference.py

Pipeline: dirty source audio (video, music in background, multiple
speakers) → clean voice sample for Qwen3-TTS voice cloning.

Steps:
1. ffmpeg-extract  (video → mono 44.1 kHz WAV)
2. Demucs separate (music out, vocals only)
3. Silero-VAD + Resemblyzer + KMeans clustering
4. INTERACTIVE: pick which cluster is the target speaker
5. Concat with cross-fades to target duration

```bash
# Recommended — automatically writes voices/Anna/Anna.wav:
python scripts/clean-reference.py /path/to/source.mp4 --voice Anna

# Or full output path explicitly:
python scripts/clean-reference.py /path/to/source.mp4 \
    --out voices/Anna/Anna.wav --target-duration 25
```

Work directory defaults to `/tmp/voiceagent-clean/<stem>/` — NOT inside
`voices/`. With `--keep-work` it stays after the run.

After running, create `voices/Anna/persona.txt` manually (template:
copy an existing `persona.txt` from Data/Sam/Sora and adapt it).
