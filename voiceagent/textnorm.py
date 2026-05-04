"""TTS-text normalization.

Two operations applied before sending text to TTS:
1. Numbers → German words. Decimals are read digit-by-digit after "Komma"
   (Star-Trek style: "drei Komma fünf eins" for 3.51). Plain integers use
   the natural German form ("tausendzweihundert" etc.).
2. Special characters that are not safe for speech are stripped. Allowed:
   letters incl. umlauts, whitespace, basic sentence punctuation (.,!?-:;)
   and quotes. Everything else (%, &, $, @, #, *, /, etc.) is removed.
"""
from __future__ import annotations

import re

import num2words

# Decimal: optional sign, digits, separator (.,) digits
_DEC_RX = re.compile(r"(?<!\w)(-?\d+)[.,](\d+)(?!\w)")
# Plain integer (handles thousands like 1234)
_INT_RX = re.compile(r"(?<!\w)-?\d+(?!\w)")
# Allowed character set
_ALLOWED = re.compile(r"[^\w\s.,!?\-:;«»\"'„“”äöüÄÖÜß]")
# Style-Tags entfernen — offen und schliessend (Gemma sendet manchmal
# HTML-style </style:NAME>). Bei batch-instruct werden sie vorher in
# emotion.parse() ausgewertet.
_TAG_RX = re.compile(r"</?style:[a-z_]+>")


def _digit_words_de(digits: str) -> str:
    """'51' → 'fünf eins'  (single-digit, German)."""
    return " ".join(num2words.num2words(int(d), lang="de") for d in digits)


def _expand_decimal(m: re.Match) -> str:
    whole, dec = m.group(1), m.group(2)
    sign = "minus " if whole.startswith("-") else ""
    whole = whole.lstrip("-")
    whole_word = num2words.num2words(int(whole), lang="de") if whole else "null"
    return f"{sign}{whole_word} Komma {_digit_words_de(dec)}"


def _expand_int(m: re.Match) -> str:
    s = m.group()
    sign = "minus " if s.startswith("-") else ""
    n = int(s.lstrip("-"))
    return f"{sign}{num2words.num2words(n, lang='de')}"


def normalize_for_tts(text: str) -> str:
    """Apply both passes; safe to call on any string.

    Style-Tags (<style:NAME>) werden komplett entfernt — die Loop
    splittet die Antwort vorher entlang der Tags und ruft TTS pro
    Segment einzeln auf, mit der jeweiligen Instruction.
    """
    if not text:
        return text
    text = _TAG_RX.sub("", text)
    text = _DEC_RX.sub(_expand_decimal, text)
    text = _INT_RX.sub(_expand_int, text)
    text = _ALLOWED.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text
