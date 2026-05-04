"""Emotion-Tag Parser + Mapping fuer Qwen3-TTS instruct.

LLM produziert Tags wie <style:nervous> in der Antwort. Diese werden hier
zu (segment_text, instruct_string)-Paaren geparst, sodass jedes Segment
mit eigener Instruction an Qwen3-TTS geht.

Mapping orientiert sich an Qwen3-TTS-bekannten Begriffen (cheerful, calm,
nervous etc.) — der Wrapper baut daraus einen englischen Instruction-Satz.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# Tag-Format: <style:NAME> (opening, steuert Style-Wechsel) oder
# </style:NAME> (closing, von Gemma manchmal als HTML-Schluss gesendet —
# wird aus dem Text entfernt aber loest keinen Wechsel aus).
TAG_RX = re.compile(r"</?style:[a-z_]+>")
OPEN_TAG_RX = re.compile(r"<style:([a-z_]+)>")

# Mapping <style:NAME> -> englische Instruction-Phrase fuer Qwen3-TTS.
# Begriffe stammen aus Qwen3-TTS-Dokumentation (Mood/Style/Physicality).
EMOTION_INSTRUCT: dict[str, str] = {
    # Sam-Set: schuechtern, lebhaft beim Wissen-Teilen, oft nervoes
    "nervous":      "Speak in a nervous, slightly fearful tone, breathy, with hesitations.",
    "thoughtful":   "Speak in a thoughtful, articulate tone, moderate pace.",
    "curious":      "Speak in a curious, lively tone, slightly fast-paced.",
    "excited":      "Speak in an excited, lively tone, fast-paced and breathy.",
    "apologetic":   "Speak in an apologetic, soft and hesitant tone.",
    "scared":       "Speak in a fearful, breathy and shaky tone.",
    "gentle":       "Speak in a gentle, soft tone.",
    "sad":          "Speak in a sad, soft tone, slow-paced.",
    # Data-Set: kontrolliert, sachlich, praezise
    "monotone":     "Speak in a monotone, flat and articulate voice, deliberate pace.",
    "serious":      "Speak in a serious, composed tone, deliberate.",
    "calm":         "Speak in a calm and composed tone.",
    # Optionale weitere (von keiner Persona aktiv genutzt)
    "happy":        "Speak in a cheerful and happy tone, moderate pace.",
    "warm":         "Speak in a warm, gentle tone.",
    "proud":        "Speak in a proud, forceful tone.",
    "angry":        "Speak in an angry, forceful tone.",
    "frustrated":   "Speak in a frustrated, slightly fast-paced tone.",
    "whisper":      "Speak in a whispering, breathy voice.",
}

# Default wenn Tag unbekannt oder fehlt
DEFAULT_INSTRUCT = ""


def parse(text: str) -> List[Tuple[str, str]]:
    """Parst Text mit <style:NAME>-Tags zu [(segment_text, instruct), ...].

    - Closing-Tags </style:NAME> werden vorab entfernt.
    - Opening-Tags <style:NAME> trennen Segmente und steuern den Style.
    - Text vor dem ersten Tag bekommt DEFAULT_INSTRUCT.
    - Unbekannte Tag-Namen fallen auf DEFAULT_INSTRUCT zurueck.
    - Leere Segmente werden uebersprungen.
    """
    cleaned = re.sub(r"</style:[a-z_]+>", "", text)
    parts = OPEN_TAG_RX.split(cleaned)
    # parts: [pre_text, tag1_name, txt1, tag2_name, txt2, ...]
    out: List[Tuple[str, str]] = []
    current_instruct = DEFAULT_INSTRUCT

    for i, part in enumerate(parts):
        if i == 0:
            seg = part.strip()
            if seg:
                out.append((seg, current_instruct))
        elif i % 2 == 1:
            current_instruct = EMOTION_INSTRUCT.get(part.lower(), DEFAULT_INSTRUCT)
        else:
            seg = part.strip()
            if seg:
                out.append((seg, current_instruct))
    return out


def has_tags(text: str) -> bool:
    return bool(TAG_RX.search(text))


def strip(text: str) -> str:
    """Tags raus, sonst unveraendert."""
    return TAG_RX.sub("", text)
