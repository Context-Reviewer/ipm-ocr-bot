from __future__ import annotations

import re
import unicodedata


ORE_NAMES: tuple[str, ...] = (
    "Copper",
    "Iron",
    "Lead",
    "Silica",
    "Aluminum",
    "Silver",
    "Gold",
    "Diamond",
    "Platinum",
    "Titanium",
    "Iridium",
    "Palladium",
    "Osmium",
    "Rhodium",
    "Inerton",
    "Quadium",
    "Scrith",
    "Uru",
    "Vibranium",
    "Aether",
    "Viterium",
    "Xynium",
    "Quolium",
    "Luterium",
    "Wraith",
    "Aqualite",
    "Opalite",
)

ORE_ALIASES: dict[str, str] = {
    "Aluminium": "Aluminum",
    "Silicon": "Silica",
}

PLANET_NAMES: tuple[str, ...] = (
    "Balor",
    "Drasta",
    "Anadius",
    "Dholen",
    "Verr",
    "Newton",
    "Widow",
    "Acheron",
    "Yangtze",
    "Solveig",
    "Imir",
    "Relic",
    "Nith",
    "Batalla",
    "Micah",
    "Pranas",
    "Castellus",
    "Gorgon",
    "Parnitha",
    "Orisoni",
    "Theseus",
    "Zelene",
    "Han",
    "Strennus",
    "Osun",
    "Ploitari",
    "Elysta",
    "Tikkun",
    "Satent",
    "Urla Rast",
    "Vular",
    "Nibiru",
    "Xena",
    "Rupert",
    "Pax",
    "Ivyra",
    "Utritis",
    "Dosie",
    "Zulu",
    "Unicae",
    "Dune",
    "Naraka",
    "Daedalus",
    "Clovis",
    "Zero",
    "Sotomi",
    "Remidian",
    "Muse",
    "Arabis",
    "Vesna",
    "Chandra",
    "Vega",
    "Crius",
    "Singhana",
    "Zumbia",
    "Elysium",
    "Nyota",
    "Doral",
    "Nikara",
    "Limbo",
    "Bob",
    "Midas",
    "Antigone",
    "Hecate",
    "Sterop",
    "Lavinia",
    "Ren",
    "Gorgons",
    "Pontus",
    "Leto",
    "Laconia",
    "Awohali",
    "Pegasi",
    "Typhon",
    "Surtur",
    "Vesta",
)

_PLANET_PREFIX_RE = re.compile(r"^\s*\d+\s*[\.:]\s*")
_ORE_TOKEN_RE = re.compile(r"[^A-Z]+")
_PLANET_TOKEN_RE = re.compile(r"[^A-Z]+")
_PLANET_TITLE_ALLOWED_RE = re.compile(r"^[A-Z0-9 .'-]+$")


def _ascii_upper(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return normalized.encode("ascii", "ignore").decode("ascii").upper()


def _normalize_ore_key(text: str | None) -> str:
    return _ORE_TOKEN_RE.sub("", _ascii_upper(text))


_ORE_LOOKUP: dict[str, str] = {_normalize_ore_key(name): name for name in ORE_NAMES}
_ORE_LOOKUP.update({_normalize_ore_key(alias): canonical for alias, canonical in ORE_ALIASES.items()})


def normalize_ore_name(text: str | None) -> str:
    return _ORE_LOOKUP.get(_normalize_ore_key(text), "")


def is_known_ore_name(text: str | None) -> bool:
    return bool(normalize_ore_name(text))


def strip_planet_title_decoration(text: str | None) -> str:
    candidate = _PLANET_PREFIX_RE.sub("", _ascii_upper(text))
    candidate = re.sub(r"\s+", " ", candidate).strip(" .:-'")
    return candidate.strip()


def _normalize_planet_key(text: str | None) -> str:
    return _PLANET_TOKEN_RE.sub("", _ascii_upper(text))


_PLANET_LOOKUP: dict[str, str] = {_normalize_planet_key(name): name for name in PLANET_NAMES}


def normalize_planet_name(text: str | None) -> str:
    return _PLANET_LOOKUP.get(_normalize_planet_key(strip_planet_title_decoration(text)), "")


def is_plausible_planet_title(text: str | None) -> bool:
    candidate = strip_planet_title_decoration(text)
    if not candidate:
        return False
    if normalize_planet_name(candidate):
        return True
    if not _PLANET_TITLE_ALLOWED_RE.fullmatch(candidate):
        return False
    words = [word for word in candidate.split() if word]
    if not words or len(words) > 3:
        return False
    compact = "".join(words)
    return 3 <= len(compact) <= 24

