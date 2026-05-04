"""Silero VAD wrapper. Tiny CPU model — used for turn-taking only.

Feeds 30 ms chunks (480 samples @ 16 kHz) and returns an utterance once
silence persists for `min_silence_ms`.
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import torch

from .config import VADConfig

log = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
VAD_FRAME = 512  # silero expects 512 samples @ 16 kHz


class VAD:
    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        log.info("Loading Silero VAD")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        self.model.eval()

    def is_speech(self, frame: np.ndarray) -> float:
        """Return speech probability in [0, 1] for one 512-sample frame."""
        if frame.shape[0] != VAD_FRAME:
            raise ValueError(f"VAD expects {VAD_FRAME} samples, got {frame.shape[0]}")
        with torch.no_grad():
            t = torch.from_numpy(frame.astype(np.float32))
            return float(self.model(t, VAD_SAMPLE_RATE).item())

    def utterances(self, frames: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Consume a stream of 512-sample frames, yield complete utterances."""
        in_speech = False
        buf: list[np.ndarray] = []
        silence_frames = 0

        ms_per_frame = VAD_FRAME / VAD_SAMPLE_RATE * 1000.0
        min_sil_frames = max(1, int(self.cfg.min_silence_ms / ms_per_frame))
        min_speech_frames = max(1, int(self.cfg.min_speech_ms / ms_per_frame))

        for frame in frames:
            p = self.is_speech(frame)
            if p >= self.cfg.threshold:
                in_speech = True
                buf.append(frame)
                silence_frames = 0
            elif in_speech:
                buf.append(frame)
                silence_frames += 1
                if silence_frames >= min_sil_frames:
                    speech_frames = len(buf) - silence_frames
                    if speech_frames >= min_speech_frames:
                        yield np.concatenate(buf)
                    buf = []
                    in_speech = False
                    silence_frames = 0
