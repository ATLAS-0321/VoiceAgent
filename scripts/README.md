# Scripts

Shell- und Python-Helfer fuer Setup, Start, Build und Tooling. Alle
Scripts sind ausfuehrbar (`chmod +x`) und nutzen `env.sh` fuer ROCm-Vars.

## Uebersicht

| Datei | Zweck |
|---|---|
| `start.sh` | Loop starten (lädt env + venv, ruft `python -m voiceagent`) |
| `env.sh` | ROCm/HIP Env-Vars (gfx1100, MIOPEN-Cache) — wird von allen anderen Scripts gesourced |
| `make-modelfile.sh` | Erstellt das Custom-Ollama-Modell `gemma4-data` aus `Modelfile.gemma4-data` |
| `Modelfile.gemma4-data` | Ollama-Modelfile (Sampling-Params, Gemma-4-Prompt-Format, no-thinking) — Input fuer `make-modelfile.sh` |
| `verify-rocm.sh` | Standalone Pre-Flight-Check: rocminfo + PyTorch-Import + GPU-matmul-Smoke-Test |
| `clean-reference.py` | Dirty-Source → clean Voice-Sample (Demucs + Silero-VAD + Resemblyzer/KMeans) |

## start.sh

Standard-Start des Voice-Loops:

```bash
./scripts/start.sh                       # nutzt config.yaml
./scripts/start.sh --voice Anna          # override tts.voice (subfolder in voices/)
./scripts/start.sh --list-devices        # PortAudio-Devices auflisten
./scripts/start.sh --list-voices         # verfuegbare Voices anzeigen
```

Sourct `env.sh`, aktiviert `.venv`, ruft `python -m voiceagent`.

## env.sh

ROCm/HIP-Tuning-Vars fuer gfx1100 (RX 7900 XTX). MUSS gesourced sein
bevor PyTorch-ROCm initialisiert. Wichtig:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` — RDNA3-Architektur-Override
- `MIOPEN_USER_DB_PATH` + `MIOPEN_FIND_MODE=FAST` — persistenter
  Kernel-Cache eliminiert ~12 s Cold-Start nach Reboot

Wird automatisch von `start.sh`, `verify-rocm.sh` und `install.sh`
gesourced — du musst es normalerweise nicht direkt anfassen.

## make-modelfile.sh + Modelfile.gemma4-data

Erstellt das Custom-Ollama-Modell `gemma4-data` aus dem Modelfile.
`install.sh` ruft das automatisch wenn das Modell noch nicht existiert.

Manuell ausfuehren wenn du die Sampling-Params im Modelfile
geaendert hast:

```bash
./scripts/make-modelfile.sh
```

Das Modelfile setzt nur Sampling-Params + Gemma-4-Prompt-Format —
KEINEN System-Prompt. Personas sind file-based unter `voices/<voice>/persona.txt`.

## verify-rocm.sh

Pre-Flight-Check wenn GPU rumzickt — zeigt:
- ROCm-Version + GPU-Name
- PyTorch-Version + HIP-Variante
- 2048×2048 matmul Smoke-Test (Performance-Indikator)

```bash
./scripts/verify-rocm.sh
```

## clean-reference.py

Pipeline: dirty Source-Audio (Video, Musik im Hintergrund, mehrere
Sprecher) → clean Voice-Sample fuer Qwen3-TTS-Voice-Cloning.

Steps:
1. ffmpeg-extract  (Video → mono 44.1 kHz WAV)
2. Demucs separate (Musik raus, nur Vocals)
3. Silero-VAD + Resemblyzer + KMeans-Cluster
4. INTERAKTIV: User waehlt welcher Cluster die Zielstimme ist
5. Concat mit Cross-Fades bis target duration

```bash
# Empfohlen — legt voices/Anna/Anna.wav automatisch an:
python scripts/clean-reference.py /pfad/zu/quelle.mp4 --voice Anna

# Oder voller Output-Pfad explizit:
python scripts/clean-reference.py /pfad/zu/quelle.mp4 \
    --out voices/Anna/Anna.wav --target-duration 25
```

Work-Dir landet per default in `/tmp/voiceagent-clean/<stem>/` —
NICHT mehr im `voices/`-Ordner. Mit `--keep-work` bleibt es liegen.

Anschliessend `voices/Anna/persona.txt` von Hand anlegen (Vorlage:
existierende persona.txt aus Data/Sam/Sora kopieren).
