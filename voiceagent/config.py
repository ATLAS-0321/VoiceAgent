"""Load and validate config.yaml into a typed dataclass tree.

Voice = Verzeichnis unter voices_dir. Eine Voice enthaelt:
- <voice>.wav   (Audio-Reference fuer TTS-Voice-Clone)
- persona.txt   (LLM System-Prompt — Charakter, Tags, Regeln)
- <voice>.txt   (optional, ICL-Transkript der WAV)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


@dataclass
class STTConfig:
    model: str = "qwen3-asr"
    language: str = "de"
    backend: str = "transformers"
    compute_type: str = "float16"
    device: str = "cuda"


@dataclass
class LLMConfig:
    model: str = "gemma4-data"
    endpoint: str = "http://localhost:11434"
    think: bool = False
    max_tokens: int = 256
    temperature: float = 0.7
    keep_alive: str = "15s"
    # System-Prompt wird aus voices/<voice>/persona.txt geladen.
    system_prompt: str = ""


@dataclass
class TTSConfig:
    # voice referenziert das Verzeichnis voices_dir/<voice>/
    voice: str = ""
    voices_dir: str = "voices"
    language: str = "de"             # de | en | fr | es | it | pt | ru | ja | ko | zh
    device: str = "cuda"
    # ref_text: Transkript fuer ICL-Mode. Leer = x_vector_only_mode.
    ref_text: str = ""
    volume: float = 1.0
    # Pre-FX EQ (laeuft VOR der per-Voice FX-Chain in voiceagent/fx.py)
    eq_bass_db: float = 0.0
    eq_treble_db: float = 0.0
    eq_lowpass_hz: float = 0.0


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 250


@dataclass
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 16000
    output_sample_rate: int = 24000


@dataclass
class SystemConfig:
    log_level: str = "INFO"


@dataclass
class WakePhraseCfg:
    pattern: str
    reply: str
    voice: str | None = None       # falls gesetzt: Voice (= Persona) switchen


@dataclass
class WakeCfg:
    enabled: bool = True
    word: str = "data"
    attentive_timeout_s: float = 30.0
    language: str = "de"
    phrases: list[WakePhraseCfg] = field(default_factory=list)


@dataclass
class Config:
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    wake: WakeCfg = field(default_factory=WakeCfg)
    root: Path = field(default_factory=lambda: Path(os.environ.get(
        "VOICEAGENT_ROOT", Path(__file__).resolve().parent.parent
    )))


def _merge(dc, raw: dict):
    for k, v in (raw or {}).items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def voice_dir(root: Path, voices_dir: str, voice: str) -> Path:
    return root / voices_dir / voice


def load_persona(root: Path, voices_dir: str, voice: str) -> str:
    """Lade voices/<voice>/persona.txt — leerer String wenn nicht vorhanden."""
    if not voice:
        return ""
    fp = voice_dir(root, voices_dir, voice) / "persona.txt"
    if not fp.exists():
        return ""
    return fp.read_text(encoding="utf-8").strip()


def list_voices(root: Path, voices_dir: str = "voices") -> list[str]:
    """Return all available voice names (subfolder names mit einer .wav drin)."""
    base = root / voices_dir
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and any(d.glob("*.wav")):
            out.append(d.name)
    return out


def load(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        path = cfg.root / "config.yaml"
    path = Path(path)
    if not path.exists():
        return cfg
    raw = yaml.safe_load(path.read_text()) or {}
    _merge(cfg.stt, raw.get("stt"))
    _merge(cfg.llm, raw.get("llm"))
    _merge(cfg.tts, raw.get("tts"))
    _merge(cfg.vad, raw.get("vad"))
    _merge(cfg.audio, raw.get("audio"))
    _merge(cfg.system, raw.get("system"))
    raw_wake = raw.get("wake") or {}
    raw_phrases = raw_wake.pop("phrases", None)
    _merge(cfg.wake, raw_wake)
    if raw_phrases:
        cfg.wake.phrases = [WakePhraseCfg(**p) for p in raw_phrases]
    # System-Prompt aus Voice laden (Voice = Persona-Quelle)
    cfg.llm.system_prompt = load_persona(cfg.root, cfg.tts.voices_dir, cfg.tts.voice)
    return cfg
