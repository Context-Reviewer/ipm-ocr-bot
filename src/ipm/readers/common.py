from __future__ import annotations

import re
from typing import Optional


_COMPACT_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)([KMBT]?)", re.IGNORECASE)
_ALPHA_LABEL_RE = re.compile(r"[A-Z][A-Z]+", re.IGNORECASE)
_SUFFIX = {
    "": 1,
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}


def parse_int(text: str | None) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\d+", str(text))
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def parse_compact_number(text: str | None) -> Optional[int]:
    if not text:
        return None
    raw = str(text).upper()
    matches = list(_COMPACT_RE.finditer(raw))
    if not matches:
        return None
    match = matches[-1]
    number_text = match.group(1).replace(",", "").replace(" ", "")
    suffix = match.group(2).upper()
    try:
        value = float(number_text)
    except Exception:
        return None
    return int(round(value * _SUFFIX.get(suffix, 1)))


def parse_alpha_label(text: str | None) -> str:
    if not text:
        return ""
    cleaned = str(text).replace("|", " ").replace("/", " ")
    for match in _ALPHA_LABEL_RE.finditer(cleaned):
        token = match.group(0).strip()
        if len(token) >= 3:
            return token.capitalize()
    return ""
