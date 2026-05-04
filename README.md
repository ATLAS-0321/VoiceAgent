# VoiceAgent

Lokaler deutschsprachiger Voice-Chat mit Voice-Cloning, Emotion-Steuerung
und per-Voice Audio-Effekten. Komplett offline auf AMD ROCm oder NVIDIA CUDA.

```
mic → VAD → STT (Qwen3-ASR) → Wake-Gate → LLM (Gemma 4) → TTS (Qwen3-TTS) → FX → speaker
```

## Was es kann

- **Wake-Word "data"** triggert Hörbereitschaft. Sprich "**hey data**", "**hey sam**"
  oder "**hey sora**" um direkt zwischen Voices (= Stimme + Persona) zu wechseln.
- **Drei mitgelieferte Voices**, jede mit eigener Stimme + Charakter + optional FX:
  - **Data** (Star Trek TNG) — sachlich, formell, "Sir"
  - **Samwell** (Game of Thrones) — schüchtern, gelehrt, "Mylord"
  - **Sora/Zora** (Star Trek Discovery) — empathische Schiffs-KI mit
    Sci-Fi-Audio-Effekten (Convolution-Reverb, Bitcrush, Resonanz-Peaks)
- **Emotion-Tags** im LLM-Output (`<style:nervous>`, `<style:warm>`, ...)
  steuern die TTS-Sprechweise — pro Antwort werden mehrere emotionale
  Segmente in einem einzigen GPU-Pass gerendert.
- **Conversation-Mode**: nach einer Antwort bleibt das System 10 s lang
  aufmerksam, du kannst direkt weitersprechen ohne Wake-Word.
- **Per-Voice Live-FX-Pipeline** via ffmpeg-Subprocess (z.B. Sora's
  Sci-Fi-Sound — Convolution-Reverb, Bitcrush, Vibrato, Tremolo).

## Hardware

- **GPU**: AMD (ROCm 7.x, gfx1100 getestet) ODER NVIDIA (CUDA 12.4+)
- **OS**: Linux (Ubuntu/Debian/Arch — install.sh detektiert automatisch)
- **Tools**: ffmpeg, Ollama mit `gemma4:e4b`

VRAM-Auslastung im Loop: ~10.5 GB (TTS 3 GB + ASR 1.5 GB + Gemma 6 GB).

## Installation

```bash
git clone <repo> ~/VoiceAgent
cd ~/VoiceAgent
./install.sh
```

`install.sh` ist ein interaktives TUI (über `gum`):
- Erkennt Distro (Debian/Arch) + GPU (AMD/NVIDIA)
- Installiert System-Pakete, ROCm/NVIDIA-Driver (mit Bestätigung), Ollama,
  PyTorch (passend zur GPU), Python-Deps
- Bietet bei Frischinstallation **Storage-Auswahl**: `.venv` (~15 GB) und
  `models/` (~9 GB) können auf eine andere Disk gelegt werden (Symlinks
  im Repo zeigen dorthin) — listet automatisch alle Mounts mit ≥30 GB frei
- Bei Re-Run mit lokal installierten Daten: bietet Verschieben auf andere Disk an
- Bei "Nein" auf einzelne Steps: skip + sammelt Manual-Anleitungen, am
  Ende ausgegeben

## Start

```bash
./scripts/start.sh
```

Sprich "data" (warten auf "Sir?") und dann deine Frage. Oder direkt
"hey data, was ist die Lichtgeschwindigkeit?" — Wake + Befehl in einem.

Voice für eine Session überschreiben:

```bash
./scripts/start.sh --voice Sora_Sample-Set
./scripts/start.sh --list-voices         # alle verfügbaren Voices
./scripts/start.sh --list-devices        # PortAudio-Devices
```

Oder zur Laufzeit via Wake-Phrase:

| Sage… | Voice (= Stimme + Persona) | Erste Antwort |
|---|---|---|
| `hey data` | Data_Sample-Set | "Ja, Sir." |
| `hey sam` | Samwell_Sample-Set | "Ja, Mylord." |
| `hey sora` | Sora_Sample-Set | "Ja, Captain." |

## Konfiguration

Alles in `config.yaml` (Repo-Root):

```yaml
stt:
  language: de       # de | en | fr | es | it | ...
llm:
  model: gemma4-data
  temperature: 0.6
tts:
  voice: Data_Sample-Set   # Subfolder unter voices/
  voices_dir: voices
  volume: 0.5
wake:
  enabled: true
  attentive_timeout_s: 10
```

Änderungen erfordern Neustart.

## Voices: Struktur

Eine Voice = ein Subfolder unter `voices/` mit allem zusammengehörigen:

```
voices/
├── Data_Sample-Set/
│   ├── Data_Sample-Set.wav    # Voice-Reference (mono, 10-30 s)
│   ├── persona.txt            # LLM System-Prompt (Charakter + Tags + Regeln)
│   └── Data_Sample-Set.txt    # optional: ICL-Transkript der WAV
├── Samwell_Sample-Set/
│   └── …
└── Sora_Sample-Set/
    └── …
```

## Eigene Voice hinzufügen

1. **Ordner anlegen** unter `voices/<Name>/`
2. **Sample** als `voices/<Name>/<Name>.wav` (10–30 s, mono, sauber).
   Wenn du nur dirty Source hast (Video, mit Musik / mehreren Sprechern):
   ```bash
   python scripts/clean-reference.py /pfad/zu/quelle.mp4 --voice <Name>
   ```
   erzeugt automatisch `voices/<Name>/<Name>.wav`.
3. **`voices/<Name>/persona.txt`** anlegen — Charakter, Anrede, Emotion-Tags,
   Regeln. Vorlage: bestehende `persona.txt` aus `voices/Data_Sample-Set/` etc.
4. (Optional) **Wake-Phrase** in `config.yaml` ergänzen:
   ```yaml
   wake:
     phrases:
       - pattern: "hey,?\\s+name\\b"
         reply: "Hallo."
         voice: <Name>
   ```
5. (Optional) **FX-Chain** für die Voice in `voiceagent/fx.py` hinzufügen
   (siehe `Sora_Sample-Set` als Beispiel).
6. Start: `./scripts/start.sh --voice <Name>`

## Wie's funktioniert (Kurz)

- **STT**: Qwen3-ASR-1.7B auf ROCm/CUDA, mehrsprachig
- **LLM**: Ollama mit Custom-Modelfile + `/api/generate?raw=true`, weil
  Standard-`/api/chat` immer einen `<|think|>`-Token injiziert der gemma4:e4b
  in Reasoning-Modus zwingt
- **TTS**: Qwen3-TTS-Base mit Voice-Cloning. Bei Tag-Antworten wird
  `model.generate(voice_clone_prompt + instruct_ids)` direkt aufgerufen —
  der High-Level-Wrapper hat keinen `instruct`-Parameter, aber das
  Low-Level-API kombiniert beides in einem Pass
- **FX**: ffmpeg-Subprocess mit stdin/stdout-Pipe, eine Chain pro Voice,
  läuft genau einmal am Ende auf dem konkatenierten Audio (sonst inkonsistente
  Pegel zwischen Segmenten)

Mehr Architektur in `CLAUDE.md`.

## Spätere Integration

VoiceAgent ist als Standalone-Loop gebaut, soll aber später als reiner
**STT+TTS-Service** in deerflow integriert werden. Das LLM bleibt dann
in deerflow's Chat-Session — Voice ist nur ein Input/Output-Modus.
