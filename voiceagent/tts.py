"""Qwen3-TTS Text-to-Speech wrapper mit Voice-Cloning, Emotion-Instruct
und optionaler per-Voice Live-FX-Pipeline.

Modell: Qwen3-TTS-12Hz-1.7B-Base (zero-shot voice cloning).
Output: 24 kHz mono float32 (oder stereo wenn Voice-FX aktiv).

Drei Generation-Pfade:
- stream(text): einfacher Voice-Clone-Render
- stream_with_instruct(text, instruct): Voice-Clone + Emotion-Instruction
- generate_batch_with_instruct(segments): mehrere Segmente mit unter-
  schiedlichen Instructions in EINEM model.generate()-Call (fuer Tag-
  basierte Emotion-Steuerung ohne Pausen zwischen Segmenten)

Reference-Audio: 3-30s, mono empfohlen. ref_text/ICL nur wenn explizit in
config gesetzt (x_vector_only_mode default).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torchaudio.functional as AF

from qwen_tts import Qwen3TTSModel

from .config import TTSConfig
from . import fx as fx_mod

log = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24000  # Qwen3-TTS native

HF_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

def _resolve_tts_model() -> str:
    """Project-local Checkpoint > HF Hub-ID (transformers nutzt System-Cache)."""
    local = Path(__file__).resolve().parent.parent / "models" / "qwen3-tts-base"
    if local.exists() and (local / "config.json").exists():
        return str(local)
    return HF_MODEL_ID


class TTS:
    def __init__(self, cfg: TTSConfig, project_root: Path):
        self.cfg = cfg
        self.root = project_root

        model_src = _resolve_tts_model()
        log.info("Loading Qwen3-TTS-Base from %s on %s", model_src, cfg.device)
        self.model = Qwen3TTSModel.from_pretrained(
            model_src,
            device_map=cfg.device,
            dtype=torch.bfloat16,
            attn_implementation="eager",  # FA2 may not be built for ROCm here
        )

        self.fx: Optional[fx_mod.VoiceFX] = fx_mod.for_voice(cfg.voice)
        if self.fx:
            log.info("Voice-FX enabled for %r", cfg.voice)

        self.audio_prompt_path: Optional[str] = self._resolve_voice(cfg.voice)
        # x_vector_only_mode (kein ref_text) = saubere Stimme aus Sample,
        # ohne dass das Modell die Sprechpausen/Stotterer aus dem
        # Transkript imitiert. ICL-Mode ist nur aktiv wenn cfg.ref_text
        # explizit gesetzt ist.
        self.ref_text: str = (cfg.ref_text or "").strip()
        # Cached reusable prompt (built once per voice change)
        self._prompt = None
        if self.audio_prompt_path:
            self._build_prompt()
            log.info("Voice-cloning reference: %s", self.audio_prompt_path)
        else:
            log.warning("No voice reference set — Qwen3-TTS-Base needs one. "
                        "Set tts.voice in config.yaml.")

    def _resolve_voice(self, voice: str) -> Optional[str]:
        """Sucht voices_dir/<voice>/<voice>.wav (Konvention)."""
        if not voice:
            return None
        base = self.root / self.cfg.voices_dir / voice
        if not base.is_dir():
            log.warning("Voice directory not found: %s", base)
            return None
        exts = (".wav", ".mp3", ".flac", ".ogg")
        for ext in exts:
            p = base / f"{voice}{ext}"
            if p.exists():
                return str(p)
        # Fallback: erste Audio-Datei im Voice-Ordner
        for ext in exts:
            for p in base.glob(f"*{ext}"):
                return str(p)
        log.warning("No audio file in voice directory: %s", base)
        return None

    def _build_prompt(self):
        """Pre-extract speaker features so each utterance reuses them."""
        if not self.audio_prompt_path:
            return
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=self.audio_prompt_path,
            ref_text=self.ref_text or None,
            x_vector_only_mode=not bool(self.ref_text),
        )

    def set_voice(self, voice: str) -> bool:
        path = self._resolve_voice(voice)
        if not path:
            return False
        self.audio_prompt_path = path
        # Voice-Wechsel laesst x_vector_only_mode aktiv — kein Auto-ICL.
        self.ref_text = ""
        self._build_prompt()
        # FX fuer neue Voice (de-)aktivieren
        self.fx = fx_mod.for_voice(voice)
        if self.fx:
            log.info("Voice-FX enabled for %r", voice)
        else:
            log.info("Voice-FX disabled for %r", voice)
        return True

    def _eq(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if self.cfg.eq_bass_db != 0.0:
            audio = AF.bass_biquad(audio, sr, gain=float(self.cfg.eq_bass_db))
        if self.cfg.eq_treble_db != 0.0:
            audio = AF.treble_biquad(audio, sr, gain=float(self.cfg.eq_treble_db))
        if self.cfg.eq_lowpass_hz > 0:
            audio = AF.lowpass_biquad(audio, sr, cutoff_freq=float(self.cfg.eq_lowpass_hz))
        return audio.squeeze(0)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield mono float32 audio chunks at 24 kHz.

        NOTE: qwen-tts 0.1.1 has NO output-streaming in the Python package
        (only via vLLM server). We do full generation and yield one chunk.
        """
        if not text.strip() or self._prompt is None:
            return

        gain = float(self.cfg.volume)
        wavs, _sr = self.model.generate_voice_clone(
            text=text,
            language=self._qwen_language(),
            voice_clone_prompt=self._prompt,
        )
        if not wavs:
            return
        arr = np.asarray(wavs[0], dtype=np.float32).squeeze()
        arr = self._postprocess(arr, gain)
        if arr.size > 0:
            yield arr

    def generate_batch_with_instruct(self, segments: list[tuple[str, str]]) -> np.ndarray:
        """Batch-Generation: alle Segmente in EINEM model.generate-Call,
        dann concatenated zurueck. Eliminiert Pausen zwischen Segmenten,
        weil Audio in einem Stueck an den Speaker geht.

        segments: [(text, instruct_string), ...]
        Belegt im Source: input_ids, instruct_ids, voice_clone_prompt sind
        alle list-shaped; prompt_items werden auf len(texts) repliziert.
        """
        if not segments or self._prompt is None:
            return np.zeros(0, dtype=np.float32)

        m = self.model
        underlying = m.model
        language = self._qwen_language()
        N = len(segments)

        texts = [s[0] for s in segments]
        instructs = [s[1] for s in segments]
        languages = [language] * N

        prompt_items = list(self._prompt) * N
        voice_clone_prompt_dict = m._prompt_items_to_voice_clone_prompt(prompt_items)

        input_ids = m._tokenize_texts([m._build_assistant_text(t) for t in texts])
        ref_ids = [None] * N

        instruct_ids: list = []
        for ins in instructs:
            if not ins:
                instruct_ids.append(None)
            else:
                instruct_ids.append(m._tokenize_texts([m._build_instruct_text(ins)])[0])

        gen_kwargs = m._merge_generate_kwargs()

        with torch.no_grad():
            talker_codes_list, _ = underlying.generate(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt_dict,
                instruct_ids=instruct_ids,
                languages=languages,
                non_streaming_mode=False,
                **gen_kwargs,
            )

        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, _fs = underlying.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_for_decode]
        )

        # Pro Segment nur EQ (kein FX, kein Volume).
        mono_segments: list[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wav = wav[cut:]
            arr = np.asarray(wav, dtype=np.float32).squeeze()
            arr = self._eq_only(arr)
            if arr.size > 0:
                mono_segments.append(arr)

        if not mono_segments:
            return np.zeros(0, dtype=np.float32)
        # Konkatenieren ZUERST, dann FX-Chain genau einmal — sonst loudnorm,
        # vibrato/tremolo, reverb-tail je Segment unabhaengig => Bug.
        full_mono = np.concatenate(mono_segments)
        return self._finalize_with_fx(full_mono, float(self.cfg.volume))

    def stream_with_instruct(self, text: str, instruct: str) -> Iterator[np.ndarray]:
        """Voice-Clone + Emotion-Instruct in einem Generate-Call.

        Umgeht den High-Level-Wrapper `generate_voice_clone` (der hat keinen
        instruct-Parameter) und ruft direkt `model.model.generate(...)` auf.
        Das Low-Level-API akzeptiert `voice_clone_prompt` und `instruct_ids`
        als unabhaengige Parameter — beides wird im Talker kombiniert.

        Wenn instruct leer ist, faellt es auf den normalen voice_clone-Pfad
        zurueck (kein Overhead).
        """
        if not text.strip() or self._prompt is None:
            return
        if not instruct:
            yield from self.stream(text)
            return

        m = self.model              # Qwen3TTSModel wrapper (helpers)
        underlying = m.model        # Qwen3TTSForConditionalGeneration

        language = self._qwen_language()
        languages = [language]

        prompt_items = self._prompt
        voice_clone_prompt_dict = m._prompt_items_to_voice_clone_prompt(prompt_items)
        ref_texts_for_ids = [it.ref_text for it in prompt_items]

        input_ids = m._tokenize_texts([m._build_assistant_text(text)])

        # ref_ids werden nur in ICL-Mode (nicht-leerer ref_text) befuellt
        ref_ids = []
        for rt in ref_texts_for_ids:
            if rt is None or rt == "":
                ref_ids.append(None)
            else:
                ref_ids.append(m._tokenize_texts([m._build_ref_text(rt)])[0])

        instruct_ids = [m._tokenize_texts([m._build_instruct_text(instruct)])[0]]

        gen_kwargs = m._merge_generate_kwargs()

        with torch.no_grad():
            talker_codes_list, _ = underlying.generate(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt_dict,
                instruct_ids=instruct_ids,
                languages=languages,
                non_streaming_mode=False,
                **gen_kwargs,
            )

        # Decode (analog zum Wrapper) inkl. ref_code-Praefix-Abschnitt-Cut
        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, _fs = underlying.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_for_decode]
        )
        if not wavs_all:
            return

        wav = wavs_all[0]
        if ref_code_list is not None and ref_code_list[0] is not None:
            ref_len = int(ref_code_list[0].shape[0])
            total_len = int(codes_for_decode[0].shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]

        arr = np.asarray(wav, dtype=np.float32).squeeze()
        arr = self._postprocess(arr, float(self.cfg.volume))
        if arr.size > 0:
            yield arr

    def _eq_only(self, arr: np.ndarray) -> np.ndarray:
        """Nur EQ, KEIN FX, KEIN Volume. Pro Segment OK aufrufbar.

        Wird genutzt damit FX nur EINMAL auf dem fertig konkatenierten
        Audio laeuft — sonst wuerden loudnorm/vibrato/tremolo/reverb pro
        Segment unabhaengig arbeiten und der Klang waere inkonsistent.
        """
        if arr.size == 0:
            return arr
        t = torch.from_numpy(arr)
        t = self._eq(t, OUTPUT_SAMPLE_RATE)
        return t.numpy().astype(np.float32)

    def _finalize_with_fx(self, mono: np.ndarray, gain: float) -> np.ndarray:
        """FX-Chain (1x am Ende) + Volume. Returned mono ODER stereo
        je nach FX-Pfad."""
        if mono.size == 0:
            return mono
        if self.fx is not None:
            # FX kriegt Vollpegel; cfg.volume erst NACH der Chain anwenden,
            # sonst pumpt loudnorm das leise Signal hoch und verstaerkt
            # Quantisierungsrauschen aus Bitcrusher + EQ-Boosts.
            out = self.fx.apply(mono)   # mono -> stereo
        else:
            out = mono
        return out * gain

    def _postprocess(self, arr: np.ndarray, gain: float) -> np.ndarray:
        """Single-Segment-Convenience: EQ + FX + Volume in einem Aufruf."""
        return self._finalize_with_fx(self._eq_only(arr), gain)

    def _qwen_language(self) -> str:
        """Map ISO short codes (de, en, ...) to Qwen3-TTS language strings."""
        m = {"de": "German", "en": "English", "fr": "French", "es": "Spanish",
             "it": "Italian", "pt": "Portuguese", "ru": "Russian",
             "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
        return m.get(self.cfg.language, self.cfg.language)

    def synth_full(self, text: str) -> np.ndarray:
        parts = list(self.stream(text))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)
