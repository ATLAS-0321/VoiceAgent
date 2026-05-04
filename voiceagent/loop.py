"""Realtime voice loop: mic -> VAD -> STT -> wake-gate -> LLM -> TTS -> speaker."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

from .audio import MicStream, SpeakerStream
from .config import Config
from . import emotion
from .llm import Message, OllamaLLM
from .stt import STT
from .textnorm import normalize_for_tts
from .tts import TTS
from .vad import VAD
from .wake import WakeConfig, WakeManager, State

log = logging.getLogger(__name__)


class VoiceLoop:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm = OllamaLLM(cfg.llm)
        self.stt = STT(cfg.stt)
        self.tts = TTS(cfg.tts, cfg.root)
        self.vad = VAD(cfg.vad)
        self._mic = None  # set by run() so _speak() can mute it
        from .wake import WakePhrase
        wake_phrases = [
            WakePhrase(pattern=p.pattern, reply=p.reply, voice=p.voice)
            for p in cfg.wake.phrases
        ]
        self.wake = WakeManager(WakeConfig(
            enabled=cfg.wake.enabled,
            word=cfg.wake.word,
            attentive_timeout_s=cfg.wake.attentive_timeout_s,
            language=cfg.wake.language,
            phrases=wake_phrases,
        ))
        self.history: deque[Message] = deque(maxlen=20)

    def run(self):
        if self.wake.cfg.enabled:
            log.info("Voice loop ready. Wake word: %r — say it to activate.", self.wake.cfg.word)
        else:
            log.info("Voice loop ready (wake-word disabled — every utterance goes to LLM).")

        with MicStream(self.cfg.audio) as mic, SpeakerStream(self.cfg.audio) as spk:
            self._mic = mic
            for utterance in self.vad.utterances(mic.frames()):
                t_utt = time.perf_counter()
                state = self.wake.s.state.value
                log.info("[%s] utterance: %d samples", state, len(utterance))

                text = self.stt.transcribe(utterance, sample_rate=16000)
                t_stt = time.perf_counter()
                if not text:
                    log.info("(no speech transcribed)")
                    continue

                print(f"\nYOU [{state}]: {text}", flush=True)
                log.info("YOU [%s]: %r", state, text)
                trig = self.wake.handle(text)

                if trig.ignored:
                    log.info("ignored (no wake match) for: %r", text)
                    continue

                spk.flush()
                t_first_audio = 0.0

                # Voice-Switch (laedt auch automatisch die persona.txt aus
                # dem Voice-Ordner). Done BEFORE speak_first so der Prompt
                # in der neuen Stimme kommt.
                if trig.switch_voice:
                    voice = trig.switch_voice
                    if self.tts.set_voice(voice):
                        log.info("TTS voice switched to %r", voice)
                    else:
                        log.warning("TTS voice %r not available", voice)
                    if self.llm.set_persona_from_voice(
                            self.cfg.root, self.cfg.tts.voices_dir, voice):
                        self.history.clear()  # neue Persona, frischer Kontext

                # Optional: speak a fixed phrase first (e.g. "Sir?", dismiss outro)
                if trig.speak_first:
                    print(f"BOT: {trig.speak_first}")
                    # If we just entered ATTENTIVE (wake trigger without
                    # immediate command), pre-warm the LLM in parallel so the
                    # next real request returns fast.
                    if (self.wake.s.state.value == "attentive"
                            and not trig.forward_to_llm):
                        threading.Thread(target=self.llm.warmup, daemon=True).start()
                    t_first_audio = self._say_blocking(spk, trig.speak_first)

                # Then: forward to LLM if there is a real command
                if trig.forward_to_llm:
                    self.history.append(Message("user", trig.forward_to_llm))
                    full_reply, t_first_audio = self._stream_response(
                        spk, t_utt, t_first_audio
                    )
                    if full_reply:
                        self.history.append(Message("assistant", full_reply))
                        print(f"BOT: {full_reply}")

                t_done = time.perf_counter()
                print(f"  [stt {t_stt - t_utt:.2f}s | "
                      f"first audio {t_first_audio - t_stt:.2f}s | "
                      f"total {t_done - t_utt:.2f}s | "
                      f"state -> {self.wake.s.state.value}]\n")
                spk.wait_drained()
                # small post-tail buffer so the speaker hardware fully clears
                time.sleep(0.15)
                # drain any frames captured during mute, then unmute
                while not mic._q.empty():
                    try: mic._q.get_nowait()
                    except Exception: break
                mic.set_mute(False)

                if trig.end_session:
                    # explicit dismiss: drop conversational history
                    self.history.clear()
                elif self.wake.s.state.value == "attentive":
                    # Reset attention timer AFTER TTS finished playing, so
                    # the user has the full timeout window from when they
                    # CAN actually start speaking — not from the wake event.
                    self.wake.s.entered_attentive_at = time.time()
                    log.info("[conversation] still listening for %ds ...",
                             int(self.cfg.wake.attentive_timeout_s))

    def _say_blocking(self, spk: SpeakerStream, text: str) -> float:
        """Speak a short fixed phrase, return the wall-clock time of first audio."""
        t_first = 0.0
        for chunk in self.tts.stream(text):
            if t_first == 0.0:
                t_first = time.perf_counter()
            spk.play(chunk)
        return t_first or time.perf_counter()

    def _stream_response(self, spk: SpeakerStream, t_start: float,
                         t_first_audio: float) -> tuple[str, float]:
        """Wartet auf die komplette LLM-Antwort, dann TTS.

        Wenn die Antwort <style:NAME>-Tags enthaelt, wird sie pro Segment
        einzeln generiert mit eigener Instruction (echte Emotion-Steuerung
        via instruct_ids). Sonst: alles am Stueck.
        """
        if self._mic is not None:
            self._mic.set_mute(True)

        full: list[str] = []
        for tok in self.llm.chat_stream(list(self.history)):
            full.append(tok)

        reply = "".join(full).strip()

        if emotion.has_tags(reply):
            raw_segments = emotion.parse(reply)
            normalized = [(normalize_for_tts(t), i) for t, i in raw_segments]
            normalized = [(t, i) for t, i in normalized if t]
            if normalized:
                log.info("emotion segments (batched): %d", len(normalized))
                for _, ins in normalized:
                    log.info("  -> instruct=%r", ins or "(none)")
                wav = self.tts.generate_batch_with_instruct(normalized)
                if wav.size > 0:
                    t_first_audio = time.perf_counter()
                    log.debug("TTS batch ready after %.2fs", t_first_audio - t_start)
                    spk.play(wav)
            return reply, t_first_audio or time.perf_counter()

        text = normalize_for_tts(reply)
        if not text:
            return reply, t_first_audio or time.perf_counter()

        for chunk in self.tts.stream(text):
            if t_first_audio == 0.0:
                t_first_audio = time.perf_counter()
                log.debug("TTS first chunk after %.2fs", t_first_audio - t_start)
            spk.play(chunk)

        return reply, t_first_audio or time.perf_counter()

    def _speak(self, spk: SpeakerStream, text: str,
               t_first_audio: float, t_start: float) -> float:
        text = normalize_for_tts(text.strip())
        if not text:
            return t_first_audio
        if self._mic is not None:
            self._mic.set_mute(True)
        for chunk in self.tts.stream(text):
            if t_first_audio == 0.0:
                t_first_audio = time.perf_counter()
                log.debug("TTS first chunk after %.2fs", t_first_audio - t_start)
            spk.play(chunk)
        return t_first_audio

    def close(self):
        self.llm.close()
