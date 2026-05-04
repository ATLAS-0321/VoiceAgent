"""Microphone capture + speaker playback via PortAudio (sounddevice).

Mic delivers 16 kHz mono float32 frames sized for the VAD (512 samples).
Speaker accepts 24 kHz mono float32 chunks from the TTS.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator

import numpy as np
import sounddevice as sd

from .config import AudioConfig
from .vad import VAD_FRAME, VAD_SAMPLE_RATE

log = logging.getLogger(__name__)


class MicStream:
    """Continuous mic capture as 512-sample frames at 16 kHz mono.

    Supports half-duplex muting via `set_mute(True)` — incoming frames are
    dropped while the speaker is playing back, preventing the agent from
    hearing its own TTS output.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._muted = False

    def set_mute(self, muted: bool):
        if muted != self._muted:
            log.debug("mic %s", "MUTED" if muted else "UNMUTED")
        self._muted = muted

    def _callback(self, indata, frames, time, status):
        if status:
            log.debug("mic status: %s", status)
        if self._muted:
            return
        self._q.put(indata[:, 0].copy())

    def __enter__(self):
        self._stream = sd.InputStream(
            samplerate=VAD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_FRAME,
            device=self.cfg.input_device,
            callback=self._callback,
        )
        self._stream.start()
        log.info("Mic open: device=%s sr=%d block=%d",
                 self.cfg.input_device, VAD_SAMPLE_RATE, VAD_FRAME)
        return self

    def __exit__(self, *_):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            yield self._q.get()


class SpeakerStream:
    """Speaker output at 24 kHz STEREO. Accepts mono (1D) or stereo (2D
    shape (samples, 2)) float32 chunks — mono wird automatisch in beide
    Kanaele dupliziert."""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        # Buffer ist immer 2D (samples, 2)
        self._buf = np.zeros((0, 2), dtype=np.float32)
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None

    def _callback(self, outdata, frames, time, status):
        if status:
            log.debug("spk status: %s", status)
        with self._lock:
            n = min(len(self._buf), frames)
            outdata[:n, :] = self._buf[:n, :]
            self._buf = self._buf[n:, :]
        if n < frames:
            outdata[n:, :] = 0.0

    def is_playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def __enter__(self):
        self._stream = sd.OutputStream(
            samplerate=self.cfg.output_sample_rate,
            channels=2,
            dtype="float32",
            blocksize=0,
            device=self.cfg.output_device,
            callback=self._callback,
        )
        self._stream.start()
        log.info("Speaker open: device=%s sr=%d ch=2",
                 self.cfg.output_device, self.cfg.output_sample_rate)
        return self

    def __exit__(self, *_):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def play(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            # Mono → in beide Kanaele duplizieren
            chunk = np.stack([chunk, chunk], axis=1)
        elif chunk.ndim == 2 and chunk.shape[1] == 1:
            chunk = np.repeat(chunk, 2, axis=1)
        with self._lock:
            self._buf = np.concatenate([self._buf, chunk], axis=0)

    def wait_drained(self):
        import time as _t
        while True:
            with self._lock:
                if len(self._buf) == 0:
                    return
            _t.sleep(0.02)

    def flush(self):
        with self._lock:
            self._buf = np.zeros((0, 2), dtype=np.float32)


def list_devices() -> str:
    return str(sd.query_devices())
