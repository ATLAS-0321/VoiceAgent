"""Wake-word + dismiss detection.

State machine:

    IDLE  ──"Data"────────────────────────►  ATTENTIVE
       │  (no command)        + speak "Sir?"     │
       │                                         │
       │  "Data, <command>"                      │
       └────────────────────────► (LLM reply) ◄──┘
                                                 │
       ◄─── dismiss-phrase ─── (random Data outro)
       ◄─── timeout ──────────────────────────

In ATTENTIVE the next utterance is treated as a command without needing
the wake word again. If it matches a dismiss pattern (German + English),
a short Data-style acknowledgement is spoken and we return to IDLE.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(Enum):
    IDLE = "idle"
    ATTENTIVE = "attentive"


# --- Patterns ---------------------------------------------------------------

# Match "data" as standalone token at the start, optionally followed by
# punctuation and a command. Handles German ASR output quirks (Daten,
# Datas, etc. are NOT matched on purpose).
_WAKE_RX = re.compile(
    r"""^\s*
        (?:hey[,\s]+|computer[,\s]+|hallo[,\s]+)?  # optional address (with optional comma)
        data                                        # the wake word
        \s*[,.:;!?-]?\s*                            # optional punctuation
        (?P<rest>.*?)\s*$                           # remainder = command
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

_DISMISS_PATTERNS = [
    r"\bnicht\s+(?:mit\s+)?dir\s+(?:gesprochen|geredet|gemeint)\b",
    r"\bwar\s+nicht\s+(?:fuer|für)\s+dich\b",
    r"\bnicht\s+zu\s+dir\b",
    r"\b(?:lass|laß|laß\s+es|lass\s+es|lass\s+mal|vergiss\s+es|vergiß\s+es)\b",
    r"\bist\s+nichts\b",
    r"\bnicht(?:s)?\s+wichtig(?:es)?\b",
    r"\bschon\s+gut\b",
    r"\b(?:never\s*mind|nevermind)\b",
    r"\bnot\s+(?:talking|speaking)\s+to\s+you\b",
    r"\bforget\s+(?:it|that)\b",
    r"\bsorry,?\s+not\s+you\b",
    r"\bich\s+(?:rede|spreche)\s+(?:nicht|mit\s+jemand)\s+(?:mit\s+dir|anderem)\b",
]
_DISMISS_RX = re.compile("|".join(_DISMISS_PATTERNS), re.IGNORECASE)

# Random Data-style outros after a dismiss. Kept short so they speak fast.
DISMISS_REPLIES_DE = [
    "Verstanden, Sir.",
    "Selbstverstaendlich, Sir.",
    "Wie Sie wuenschen, Sir.",
    "Notiert. Ich kehre in den Standby-Modus zurueck.",
    "Verzeihung fuer die Stoerung, Sir.",
    "Ich habe falsch interpretiert. Entschuldigung, Sir.",
    "Akzeptiert. Ich bleibe in Bereitschaft.",
]

DISMISS_REPLIES_EN = [
    "Understood, Sir.",
    "Acknowledged, Sir.",
    "Of course, Sir.",
    "Returning to standby mode.",
    "My apologies for the interruption, Sir.",
    "Noted. I will remain on standby.",
]

ATTENTIVE_PROMPTS_DE = ["Sir?", "Ja, Sir?", "Captain?", "Wie kann ich behilflich sein?"]
ATTENTIVE_PROMPTS_EN = ["Sir?", "Yes, Sir?", "Captain?", "How may I be of assistance?"]


@dataclass
class WakePhrase:
    """A wake phrase with its specific reply.

    `pattern` is a regex (case-insensitive). If it matches the whole utterance,
    `reply` is spoken back AND the loop enters ATTENTIVE waiting for command.
    If `pattern` matches a prefix and there's text after, that text becomes
    the command (no `reply` spoken first).

    Optional: voice — wenn gesetzt, switcht der Loop die Voice (= Audio-Sample
    + Persona, weil persona.txt im Voice-Ordner liegt).
    """
    pattern: str
    reply: str
    voice: Optional[str] = None


@dataclass
class WakeConfig:
    enabled: bool = True
    word: str = "data"                    # legacy, used if no phrases set
    attentive_timeout_s: float = 30.0
    language: str = "de"
    phrases: list[WakePhrase] = field(default_factory=list)


@dataclass
class WakeState:
    state: State = State.IDLE
    entered_attentive_at: float = 0.0


@dataclass
class TriggerResult:
    """What the loop should do with a transcribed utterance."""
    speak_first: Optional[str] = None       # speak this BEFORE the LLM
    forward_to_llm: Optional[str] = None    # text to send to the LLM
    end_session: bool = False               # after speaking, return to IDLE
    ignored: bool = False                   # nothing to do, drop the utterance
    switch_voice: Optional[str] = None      # if set: switch voice (and persona)


def _pick(opts: list[str]) -> str:
    return random.choice(opts)


class WakeManager:
    def __init__(self, cfg: WakeConfig):
        self.cfg = cfg
        self.s = WakeState()

    def reset(self):
        self.s = WakeState()

    def _check_timeout(self):
        if (self.s.state == State.ATTENTIVE
                and time.time() - self.s.entered_attentive_at > self.cfg.attentive_timeout_s):
            self.s = WakeState()  # auto-sleep

    def _enter_attentive(self):
        self.s.state = State.ATTENTIVE
        self.s.entered_attentive_at = time.time()

    def _outro(self) -> str:
        bank = DISMISS_REPLIES_DE if self.cfg.language.startswith("de") else DISMISS_REPLIES_EN
        return _pick(bank)

    def _attentive_prompt(self) -> str:
        bank = ATTENTIVE_PROMPTS_DE if self.cfg.language.startswith("de") else ATTENTIVE_PROMPTS_EN
        return _pick(bank)

    def _match_custom_phrase(self, text: str) -> Optional[tuple[WakePhrase, str]]:
        """Returns (phrase, rest) when matched. `rest` is the text after the
        matched phrase, leading punctuation/whitespace stripped. Empty if
        the phrase consumed the whole utterance."""
        for ph in self.cfg.phrases:
            m = re.search(ph.pattern, text, re.IGNORECASE)
            if m:
                rest = text[m.end():]
                rest = re.sub(r"^[\s,.:;!?\-]+", "", rest).strip()
                return ph, rest
        return None

    def handle(self, text: str) -> TriggerResult:
        """Decide what to do with a freshly transcribed utterance."""
        if not self.cfg.enabled:
            return TriggerResult(forward_to_llm=text)

        self._check_timeout()
        text = text.strip()
        if not text:
            return TriggerResult(ignored=True)

        # IDLE: only triggered by wake word OR custom wake phrase.
        # Custom-Phrase wird auch in ATTENTIVE geprueft, damit "hey sam"
        # mitten im Gespraech die Persona switchen kann.
        if self.s.state == State.IDLE:
            ph_match = self._match_custom_phrase(text)
            if ph_match:
                ph, rest = ph_match
                self._enter_attentive()
                if rest:
                    # "hey sam, erzaehl mir was" -> switch + sofort LLM,
                    # kein "Ja, Mylord" zwischenrein.
                    return TriggerResult(
                        forward_to_llm=rest,
                        switch_voice=ph.voice,
                    )
                return TriggerResult(
                    speak_first=ph.reply,
                    switch_voice=ph.voice,
                )

            # generic "data" wake word (built-in regex)
            m = _WAKE_RX.match(text)
            if not m:
                return TriggerResult(ignored=True)

            rest = m.group("rest").strip()
            if not rest:
                # Just "Data" alone -> attentive prompt, await command
                self._enter_attentive()
                return TriggerResult(speak_first=self._attentive_prompt())

            # "Data, <command>" -> handle the command, stay IDLE after
            return TriggerResult(forward_to_llm=rest)

        # ATTENTIVE: dismiss-Phrase oder neuer Befehl
        if _DISMISS_RX.search(text):
            self.s = WakeState()  # back to IDLE
            return TriggerResult(speak_first=self._outro(), end_session=True)

        # Voice-Switch auch im ATTENTIVE: "hey data" / "hey sam" wechselt
        # mid-conversation. Behaelt die ATTENTIVE-Window aktiv.
        ph_match = self._match_custom_phrase(text)
        if ph_match and ph_match[0].voice:
            ph, rest = ph_match
            self.s.entered_attentive_at = time.time()
            if rest:
                return TriggerResult(
                    forward_to_llm=rest,
                    switch_voice=ph.voice,
                )
            return TriggerResult(
                speak_first=ph.reply,
                switch_voice=ph.voice,
            )

        # Real command -> answer + STAY ATTENTIVE (conversation mode).
        # Reset the timeout so the user has another `attentive_timeout_s`
        # window to follow up before auto-sleeping.
        self.s.entered_attentive_at = time.time()
        return TriggerResult(forward_to_llm=text)
