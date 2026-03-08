from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .common import parse_compact_number, parse_int


_TITLE_RE = re.compile(r"^\s*(\d{1,2})\s*[\.:]\s+([A-Z][A-Z0-9 .'-]*)")
_ALPHA_RE = re.compile(r"[^A-Z]+")
_ORE_ALIASES = {
    "COPPER": "Copper",
    "IRON": "Iron",
    "LEAD": "Lead",
    "SILICA": "Silica",
    "ALUMINUM": "Aluminum",
    "ALUMINIUM": "Aluminum",
    "TITANIUM": "Titanium",
    "SILVER": "Silver",
}
_LEVEL_SKIP_TOKENS = ("SEC", "KPH", "MPH", "MKPH", "/", "RATE", "YIELD", "COST", "$")


@dataclass(slots=True, frozen=True)
class ParsedOreRow:
    ore_name: str
    quantity: int | None


@dataclass(slots=True, frozen=True)
class ParsedPlanetPanel:
    title: str = ""
    planet_id: int | None = None
    mining_level: int | None = None
    speed_level: int | None = None
    cargo_level: int | None = None
    mining_cost: int | None = None
    speed_cost: int | None = None
    cargo_cost: int | None = None


def _ascii_upper(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return normalized.encode("ascii", "ignore").decode("ascii").upper()


def _clean_lines(text: str | None) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def normalize_ore_name(text: str | None) -> str:
    token = _ALPHA_RE.sub("", _ascii_upper(text))
    return _ORE_ALIASES.get(token, "")


def _extract_level(lines: list[str], keyword: str) -> int | None:
    ascii_lines = [_ascii_upper(line) for line in lines]
    for index, ascii_line in enumerate(ascii_lines):
        if keyword not in ascii_line:
            continue
        for candidate in ascii_lines[index : index + 3]:
            if any(token in candidate for token in _LEVEL_SKIP_TOKENS):
                continue
            value = parse_int(candidate)
            if value is not None:
                return value
    return None


def parse_ore_panel_text(
    text: str | None,
    *,
    visible_rows: int,
    known_names: tuple[str, ...] | list[str] = (),
) -> list[ParsedOreRow]:
    lines = _clean_lines(text)
    if not lines:
        return []

    known_lookup = {normalize_ore_name(name): normalize_ore_name(name) for name in known_names if normalize_ore_name(name)}
    names: list[str] = []
    quantities: list[int] = []
    seen_names: set[str] = set()

    for line in lines:
        name = normalize_ore_name(line)
        if name:
            canonical = known_lookup.get(name, name)
            if canonical not in seen_names:
                names.append(canonical)
                seen_names.add(canonical)
            continue
        if "$" in line:
            continue
        quantity = parse_compact_number(line)
        if quantity is not None:
            quantities.append(quantity)

    row_count = min(max(0, int(visible_rows)), len(names), len(quantities))
    return [ParsedOreRow(ore_name=names[index], quantity=quantities[index]) for index in range(row_count)]


def parse_planet_panel_text(text: str | None) -> ParsedPlanetPanel:
    lines = _clean_lines(text)
    if not lines:
        return ParsedPlanetPanel()

    title = ""
    planet_id = None
    for line in lines:
        ascii_line = re.sub(r"\s+", " ", _ascii_upper(line)).strip()
        match = _TITLE_RE.match(ascii_line)
        if not match:
            continue
        planet_id = int(match.group(1))
        title = f"{planet_id}. {match.group(2).strip()}"
        break

    values = [value for value in (parse_compact_number(line) for line in lines) if value is not None]
    mining_cost = speed_cost = cargo_cost = None
    if len(values) >= 3:
        mining_cost, speed_cost, cargo_cost = values[-3:]

    return ParsedPlanetPanel(
        title=title,
        planet_id=planet_id,
        mining_level=_extract_level(lines, "MINING"),
        speed_level=_extract_level(lines, "SHIP"),
        cargo_level=_extract_level(lines, "CARGO"),
        mining_cost=mining_cost,
        speed_cost=speed_cost,
        cargo_cost=cargo_cost,
    )
