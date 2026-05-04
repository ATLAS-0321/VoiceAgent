"""Qwen3-ASR Speech-to-Text wrapper.

Qwen3-ASR-1.7B, multilingual (German + 25 weitere Sprachen), runs on
PyTorch-ROCm. Audio input: 16 kHz mono float32, range [-1.0, 1.0].
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from qwen_asr import Qwen3ASRModel

from .config import STTConfig

log = logging.getLogger(__name__)

HF_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

def _resolve_asr_model() -> str:
    """Project-local Checkpoint > HF Hub-ID (transformers nutzt System-Cache)."""
    local = Path(__file__).resolve().parent.parent / "models" / "qwen3-asr"
    if local.exists() and (local / "config.json").exists():
        return str(local)
    return HF_MODEL_ID


class STT:
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        device = cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("device=cuda but no GPU; falling back to CPU")
            device = "cpu"

        model_src = _resolve_asr_model()
        log.info("Loading Qwen3-ASR from %s on %s", model_src, device)
        self.model = Qwen3ASRModel.from_pretrained(
            model_src,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            max_new_tokens=256,
        )

        # Map our config language (ISO short) → Qwen3-ASR full name.
        _lang_map = {
            "de": "German", "en": "English", "fr": "French", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
            "ko": "Korean", "zh": "Chinese", "nl": "Dutch", "sv": "Swedish",
            "pl": "Polish", "cs": "Czech", "fi": "Finnish", "da": "Danish",
            "tr": "Turkish", "ar": "Arabic", "vi": "Vietnamese", "th": "Thai",
            "id": "Indonesian", "hi": "Hindi", "fa": "Persian", "el": "Greek",
            "ro": "Romanian", "hu": "Hungarian", "fil": "Filipino", "mk": "Macedonian",
            "ms": "Malay", "yue": "Cantonese",
        }
        if cfg.language in ("auto", ""):
            self._language = None
        elif cfg.language in _lang_map:
            self._language = _lang_map[cfg.language]
        else:
            # Already a full name
            self._language = cfg.language.title()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if sample_rate != 16000:
            raise ValueError(f"Qwen3-ASR expects 16 kHz, got {sample_rate}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Qwen3-ASR.transcribe accepts a (numpy, sr) tuple OR a file path
        results = self.model.transcribe(
            audio=(audio, sample_rate),
            language=self._language,
        )
        if not results:
            return ""
        return results[0].text.strip()
