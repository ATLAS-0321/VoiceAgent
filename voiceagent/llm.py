"""Ollama LLM wrapper using /api/generate with raw=true.

Why raw=true:
  Ollama's default chat templating injects the Gemma 4 `<|think|>` token,
  which forces the model into reasoning mode regardless of the API
  `think: false` flag. With raw=true we send the prompt verbatim in the
  Gemma 4 turn format WITHOUT `<|think|>`, and the model produces clean
  direct answers (no <channel|> separator, no thinking block).

  Verified: eval_count drops from 200+ to ~60 tokens for the same query,
  ~3× faster generation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterator

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str


def _gemma4_format(system: str, history: list[Message]) -> str:
    """Build a Gemma 4 raw prompt with NO <|think|> token.

    Format per ai.google.dev/gemma/docs/core/prompt-formatting-gemma4:
      <|turn>system\n...<turn|>
      <|turn>user\n...<turn|>
      <|turn>assistant\n...<turn|>
      <|turn>assistant\n     ← generation start
    """
    parts: list[str] = []
    if system:
        parts.append(f"<|turn>system\n{system}<turn|>")
    for m in history:
        role = "assistant" if m.role == "assistant" else "user"
        parts.append(f"<|turn>{role}\n{m.content}<turn|>")
    parts.append("<|turn>assistant\n")
    return "\n".join(parts)


class OllamaLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))

    def set_persona_from_voice(self, root, voices_dir: str, voice: str) -> bool:
        """Switch active persona by reloading voices/<voice>/persona.txt.
        Returns True wenn die persona.txt existiert und geladen wurde.
        """
        from .config import load_persona
        text = load_persona(root, voices_dir, voice)
        if not text:
            log.warning("persona.txt not found for voice %r", voice)
            return False
        self.cfg.system_prompt = text
        log.info("LLM persona switched to voice %r (%d chars)", voice, len(text))
        return True

    def chat_stream(self, history: list[Message]) -> Iterator[str]:
        """Yield answer tokens as they arrive."""
        prompt = _gemma4_format(self.cfg.system_prompt, history)
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_predict": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
            },
        }
        url = f"{self.cfg.endpoint}/api/generate"

        with self.client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("non-JSON chunk: %r", line[:80])
                    continue
                tok = chunk.get("response", "")
                if tok:
                    yield tok
                if chunk.get("done"):
                    break

    def chat(self, history: list[Message]) -> str:
        return "".join(self.chat_stream(history))

    def warmup(self):
        """Pre-load the model + KV cache so the first real request returns
        with minimal latency."""
        try:
            payload = {
                "model": self.cfg.model,
                "prompt": _gemma4_format(self.cfg.system_prompt, [Message("user", "hi")]),
                "raw": True,
                "stream": False,
                "keep_alive": self.cfg.keep_alive,
                "options": {"num_predict": 1, "temperature": 0.1},
            }
            self.client.post(
                f"{self.cfg.endpoint}/api/generate",
                json=payload, timeout=30.0,
            )
        except Exception as e:
            log.debug("warmup failed (non-fatal): %s", e)

    def release(self):
        """Tell Ollama to unload the model quickly (1 s keep_alive)."""
        try:
            self.client.post(
                f"{self.cfg.endpoint}/api/generate",
                json={"model": self.cfg.model, "prompt": "", "keep_alive": "1s"},
                timeout=10.0,
            )
        except Exception as e:
            log.debug("release failed (non-fatal): %s", e)

    def close(self):
        self.release()
        self.client.close()
