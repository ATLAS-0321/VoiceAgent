# VoiceAgent

Local realtime voice chat with voice cloning, emotion control and per-voice
audio effects. Fully offline on AMD ROCm or NVIDIA CUDA.

```
mic → VAD → STT (Qwen3-ASR) → wake-gate → LLM (Gemma 4) → TTS (Qwen3-TTS) → FX → speaker
```

## What it does

- **Wake word "data"** activates listening. Say "**hey data**", "**hey sam**"
  or "**hey sora**" to switch between voices (= persona + audio sample) on the fly.
- **Three bundled voices**, each with its own sample, character and optional FX:
  - **Data** (Star Trek TNG) — analytical, formal, addresses you as "Sir"
  - **Samwell** (Game of Thrones) — shy, scholarly, says "Mylord"
  - **Sora/Zora** (Star Trek Discovery) — empathetic ship AI with sci-fi
    audio effects (convolution reverb, bitcrush, resonance peaks, vibrato)
- **Emotion tags** in the LLM output (`<style:nervous>`, `<style:warm>`, …)
  steer the TTS delivery — multiple emotional segments per response render
  in a single GPU pass.
- **Conversation mode**: after each reply the system stays attentive for
  10 seconds, you can keep talking without re-saying the wake word.
- **Per-voice live FX pipeline** via ffmpeg subprocess (e.g. Sora's sci-fi
  sound — convolution reverb, bitcrush, vibrato, tremolo).

## Hardware

- **GPU**: AMD (ROCm 7.x, gfx1100 tested) OR NVIDIA (CUDA 12.4+)
- **OS**: Linux (Ubuntu/Debian/Arch — `install.sh` auto-detects)
- **Tools**: ffmpeg, Ollama with `gemma4:e4b`

VRAM at runtime: ~10.5 GB (TTS 3 GB + ASR 1.5 GB + Gemma 6 GB).

## Installation

```bash
git clone https://github.com/ATLAS-0321/VoiceAgent.git ~/VoiceAgent
cd ~/VoiceAgent
./install.sh
```

`install.sh` is an interactive TUI (via `gum`):
- Detects distro (Debian/Arch) and GPU (AMD/NVIDIA)
- Installs system packages, ROCm/NVIDIA drivers (with confirmation),
  Ollama, PyTorch (matching the GPU), Python deps
- On a fresh install offers a **storage selection**: `.venv` (~15 GB) and
  `models/` (~9 GB) can live on a separate disk (symlinks in the repo
  point to it) — automatically lists all mounts with ≥30 GB free
- On re-run with locally installed data: offers to move it to another disk
- If you decline a step: it's skipped, manual instructions are collected
  and printed at the end

## Start

```bash
./scripts/start.sh
```

Say "data" (wait for "Sir?") then your question. Or say it in one breath:
"hey data, what is the speed of light?" — wake + command combined.

Override voice for a single session:

```bash
./scripts/start.sh --voice Sora_Sample-Set
./scripts/start.sh --list-voices         # list available voices
./scripts/start.sh --list-devices        # list PortAudio devices
```

Or switch at runtime via wake phrase:

| Say… | Voice (= sample + persona) | First reply |
|---|---|---|
| `hey data` | Data_Sample-Set | "Ja, Sir." |
| `hey sam` | Samwell_Sample-Set | "Ja, Mylord." |
| `hey sora` | Sora_Sample-Set | "Ja, Captain." |

## Configuration

Everything in `config.yaml` (repo root):

```yaml
stt:
  language: de       # de | en | fr | es | it | ...
llm:
  model: gemma4-data
  temperature: 0.6
tts:
  voice: Data_Sample-Set   # subfolder under voices/
  voices_dir: voices
  volume: 0.5
wake:
  enabled: true
  attentive_timeout_s: 10
```

Changes require a restart.

## Voices: layout

A voice = one subfolder under `voices/` containing everything that
belongs together:

```
voices/
├── Data_Sample-Set/
│   ├── Data_Sample-Set.wav    # voice reference (mono, 10–30 s)
│   ├── persona.txt            # LLM system prompt (character + tags + rules)
│   └── Data_Sample-Set.txt    # optional: ICL transcript of the WAV
├── Samwell_Sample-Set/
│   └── …
└── Sora_Sample-Set/
    └── …
```

## Adding your own voice

1. **Create folder** `voices/<Name>/`
2. **Sample** as `voices/<Name>/<Name>.wav` (10–30 s, mono, clean).
   For dirty source material (video, music in background, multiple
   speakers):
   ```bash
   python scripts/clean-reference.py /path/to/source.mp4 --voice <Name>
   ```
   automatically writes to `voices/<Name>/<Name>.wav`.
3. **`voices/<Name>/persona.txt`** — character, address, emotion tags,
   rules. Copy an existing one (e.g. `voices/Data_Sample-Set/persona.txt`)
   as a template.
4. (Optional) **Wake phrase** in `config.yaml`:
   ```yaml
   wake:
     phrases:
       - pattern: "hey,?\\s+name\\b"
         reply: "Hello."
         voice: <Name>
   ```
5. (Optional) **FX chain** for the voice in `voiceagent/fx.py`
   (see `Sora_Sample-Set` for an example).
6. Run: `./scripts/start.sh --voice <Name>`

## How it works (short)

- **STT**: Qwen3-ASR-1.7B on ROCm/CUDA, multilingual
- **LLM**: Ollama with custom Modelfile + `/api/generate?raw=true`,
  because the standard `/api/chat` always injects a `<|think|>` token
  that forces gemma4:e4b into reasoning mode
- **TTS**: Qwen3-TTS-Base with voice cloning. For tagged responses we
  call `model.generate(voice_clone_prompt + instruct_ids)` directly —
  the high-level wrapper has no `instruct` parameter, but the low-level
  API combines both in a single pass
- **FX**: ffmpeg subprocess via stdin/stdout pipe, one chain per voice,
  runs exactly once on the concatenated audio at the end (otherwise the
  level/reverb/modulation differs per segment)

## Built with AI assistance

This project was developed in collaboration with an AI coding assistant
(Anthropic Claude) — design, implementation, debugging and documentation.
Architectural decisions, audio tuning and the final say belong to the
maintainer; the AI handled most of the typing and a lot of the research.

## License & contributing

MIT licensed (see `LICENSE`). Contributions welcome via pull request —
see `CONTRIBUTING.md` for the contributor agreement.

## Future integration

VoiceAgent is built as a standalone loop, but is intended to later be
integrated into deerflow as a pure **STT+TTS service**. The LLM will then
live in deerflow's chat session — voice is just an additional input/output
modality.
