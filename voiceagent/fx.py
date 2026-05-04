"""Per-Voice Live-FX-Pipeline via ffmpeg-Pipe.

Filter-Chain ist EXAKT gleich wie die Test-Variante die Atlas getuned hat:
EQ-Peaks, Aecho-Combfilter, Acrusher (Bitcrush), Convolution-Reverb mit
synthetischem IR, Loudnorm, Vibrato, Tremolo.

Audio (numpy float32 mono @ 24kHz) wird via stdin in einen ffmpeg-Subprozess
gepiped, durch eine voice-spezifische Filter-Chain prozessiert, und als
Stereo float32 @ 24kHz zurueckgelesen.

Kein File-IO fuer Audio — IR-File ist persistent im _assets/-Verzeichnis.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
ASSETS_DIR = Path(__file__).parent / "_assets"


def _ensure_sora_ir() -> Path:
    """Generates the synthetic exponential-decay reverb IR for Sora — exakt
    derselbe IR wie in der Test-Tuning-Phase (600ms decay rate 8.0)."""
    ir_path = ASSETS_DIR / "sora_reverb_ir.wav"
    if ir_path.exists():
        return ir_path
    ASSETS_DIR.mkdir(exist_ok=True)
    sr = SAMPLE_RATE
    dur = 0.6
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    env = np.exp(-t * 8.0)
    rng = np.random.default_rng(seed=42)   # deterministisch
    ir_l = rng.standard_normal(n) * env
    ir_r = rng.standard_normal(n) * env
    ir = np.stack([ir_l, ir_r], axis=1).astype(np.float32)
    ir /= np.max(np.abs(ir))
    ir *= 0.7
    sf.write(str(ir_path), ir, sr, subtype="PCM_16")
    log.info("Generated reverb IR at %s", ir_path)
    return ir_path


# Voice-spezifische FX-Definitionen.
# Jeder Eintrag: graph = filter_complex (mit [0:a] = pipe-input, optional
# [1:a] = IR-file). out_label = der finale Pad-Label fuer -map.
# Werte EXAKT wie im Atlas-getuneten Test-Befehl.
VOICE_FX: dict[str, dict] = {
    "Sora_Sample-Set": {
        "ir_path": None,    # gesetzt von _ensure_sora_ir() beim init
        "graph": (
            "[0:a]aformat=sample_rates=24000:channel_layouts=mono,"
            "pan=stereo|c0=c0|c1=c0,"
            "equalizer=f=240:t=q:w=1.0:g=1.5,"
            "equalizer=f=2500:t=q:w=1.0:g=2.0,"
            "equalizer=f=3500:t=q:w=0.4:g=3.8,"
            "equalizer=f=4500:t=q:w=0.4:g=2.5,"
            "equalizer=f=5500:t=q:w=0.4:g=3.0,"
            "equalizer=f=8500:t=q:w=1.0:g=-2,"
            "aecho=1.0:0.38:4|7:0.28|0.22,"
            "acrusher=level_in=1:level_out=1:bits=10:mix=0.30:mode=lin:aa=1[dry];"
            "[dry]asplit=2[d1][d2];"
            "[d2][1:a]afir=dry=10:wet=10:length=1[wet];"
            "[d1][wet]amix=inputs=2:weights=1.0 0.18:dropout_transition=0,"
            "loudnorm=I=-18:TP=-1.5:LRA=8,"
            "vibrato=f=5.5:d=0.04,"
            "tremolo=f=11:d=0.05[out]"
        ),
        "out_label": "[out]",
    },
}


class VoiceFX:
    """Apply a fixed filter graph to a numpy mono signal via ffmpeg.

    apply(arr_mono_f32) -> stereo float32 (samples, 2).
    """

    def __init__(self, voice: str, sample_rate: int = SAMPLE_RATE):
        cfg = VOICE_FX[voice]
        self.voice = voice
        self.sr = sample_rate
        self.graph = cfg["graph"]
        self.out_label = cfg["out_label"]
        self.ir_path: Optional[Path] = cfg["ir_path"]
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found in PATH")

    def apply(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        mono = np.ascontiguousarray(audio.squeeze().astype(np.float32))
        if mono.ndim != 1:
            mono = mono.mean(axis=1).astype(np.float32)

        # Stille-Pad am Ende, damit der Convolution-Reverb-Tail (600ms IR)
        # plus aecho komplett ausklingen koennen, statt am Audio-Ende
        # abzubrechen. Output enthaelt damit auch die Reverb-Schwaenze.
        tail_pad = np.zeros(int(self.sr * 0.7), dtype=np.float32)
        mono = np.concatenate([mono, tail_pad])

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(self.sr), "-ac", "1", "-i", "pipe:0",
        ]
        if self.ir_path is not None:
            cmd += ["-i", str(self.ir_path)]
        cmd += [
            "-filter_complex", self.graph,
            "-map", self.out_label,
            "-f", "f32le", "-ar", str(self.sr), "-ac", "2", "pipe:1",
        ]

        proc = subprocess.run(
            cmd, input=mono.tobytes(),
            capture_output=True, check=False,
        )
        if proc.returncode != 0:
            log.error("ffmpeg fx failed (rc=%d): %s",
                      proc.returncode, proc.stderr.decode("utf-8", errors="replace")[-400:])
            return np.stack([mono, mono], axis=1)

        stereo = np.frombuffer(proc.stdout, dtype=np.float32)
        if stereo.size % 2 != 0:
            stereo = stereo[:-(stereo.size % 2)]
        return stereo.reshape(-1, 2)


def for_voice(voice: str) -> Optional[VoiceFX]:
    """Returns a VoiceFX instance for the named voice if a chain is defined."""
    if voice not in VOICE_FX:
        return None
    # Sora needs the IR file
    if voice == "Sora_Sample-Set":
        VOICE_FX[voice]["ir_path"] = _ensure_sora_ir()
    return VoiceFX(voice)
