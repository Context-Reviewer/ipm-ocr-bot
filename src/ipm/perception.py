from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import io
import json
import os
import re
import tempfile
from typing import Any, Protocol

import numpy as np
from PIL import Image

try:
    import pytesseract
except Exception:
    pytesseract = None

from .domain_data import (
    ORE_NAMES,
    RESOURCE_ROW_NAMES,
    is_plausible_planet_title,
    normalize_resource_row_name,
    resource_row_name_reject_reason,
)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

try:
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage import FileAccessMode, StorageFile
except Exception:
    BitmapDecoder = None
    OcrEngine = None
    FileAccessMode = None
    StorageFile = None

_NULLABLE_COMPACT_RE = re.compile(r"^\s*\$?\s*[0-9]+(?:[.,][0-9]+)?\s*[KMBT]?\s*$", re.IGNORECASE)
_STRICT_INT_RE = re.compile(r"^\s*\d+\s*$")
_TITLE_PROSE_PATTERNS = (
    "the visible planet title is",
    "visible planet title is",
    "planet title is",
    "the title is",
    "title is",
)
_COMPACT_SUFFIX = {
    "": 1,
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}


@dataclass(slots=True, frozen=True)
class PerceptionRead:
    value: str
    backend: str
    confidence: float = 0.0


@dataclass(slots=True, frozen=True)
class OrePanelRowJSON:
    name: str
    quantity: str | None
    price: str | None


@dataclass(slots=True, frozen=True)
class OrePanelJSON:
    panel_type: str
    planet_name: str | None
    ores: tuple[OrePanelRowJSON, ...]
    backend: str = "openai"


@dataclass(slots=True, frozen=True)
class PlanetPanelUpgradesJSON:
    mining_cost: str | None
    speed_cost: str | None
    cargo_cost: str | None


@dataclass(slots=True, frozen=True)
class PlanetPanelJSON:
    panel_type: str
    planet_name: str | None
    level: str | None
    upgrades: PlanetPanelUpgradesJSON
    cash: str | None
    backend: str = "openai"


class StructuredPerceptionError(RuntimeError):
    def __init__(self, *, backend: str, panel_type: str, reason: str, raw_output: str = "") -> None:
        super().__init__(f"{backend} {panel_type} failure: {reason}")
        self.backend = backend
        self.panel_type = panel_type
        self.reason = reason
        self.raw_output = raw_output


class PerceptionBackend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def read_text(self, image: Image.Image, *, prompt: str = "", mode: str = "generic") -> PerceptionRead:
        ...


def _normalize_multiline_text(text: str | None) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_response_text(response: object) -> str:
    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return text
    parts: list[str] = []
    for output in getattr(response, "output", ()) or ():
        for content in getattr(output, "content", ()) or ():
            value = getattr(content, "text", None)
            if value:
                parts.append(str(value))
    return "\n".join(part for part in parts if part).strip()


def _validated_nullable_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value.strip() or None


def _parse_compact_number_text(text: str | None) -> int | None:
    if not text:
        return None
    cleaned = str(text).upper().replace("$", "").strip()
    match = re.fullmatch(r"([0-9]+(?:[.,][0-9]+)?)([KMBT]?)", cleaned)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except Exception:
        return None
    return int(round(value * _COMPACT_SUFFIX.get(match.group(2).upper(), 1)))


def _parse_strict_int(text: str | None) -> int | None:
    if not text or not _STRICT_INT_RE.fullmatch(str(text)):
        return None
    try:
        return int(str(text).strip())
    except Exception:
        return None

def _has_title_prose(text: str | None) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    if any(pattern in normalized for pattern in _TITLE_PROSE_PATTERNS):
        return True
    if '"' in normalized:
        return True
    if normalized.startswith(("the ", "this ")) and "title" in normalized:
        return True
    return False


def _validate_ore_panel_payload(payload: object) -> OrePanelJSON:
    if not isinstance(payload, dict):
        raise ValueError("ore_panel payload must be an object")
    expected_keys = {"panel_type", "planet_name", "ores"}
    extra_keys = set(payload.keys()) - expected_keys
    if extra_keys:
        raise ValueError(f"ore_panel payload has extra keys: {sorted(extra_keys)}")
    if payload.get("panel_type") != "ore_panel":
        raise ValueError("ore_panel payload has invalid panel_type")
    raw_ores = payload.get("ores")
    if not isinstance(raw_ores, list):
        raise ValueError("ore_panel payload ores must be an array")
    rows: list[OrePanelRowJSON] = []
    for index, entry in enumerate(raw_ores):
        if not isinstance(entry, dict):
            raise ValueError(f"ore_panel ores[{index}] must be an object")
        expected_row_keys = {"name", "quantity", "price"}
        row_extra = set(entry.keys()) - expected_row_keys
        if row_extra:
            raise ValueError(f"ore_panel ores[{index}] has extra keys: {sorted(row_extra)}")
        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError(f"ore_panel ores[{index}].name must be a string")
        rows.append(
            OrePanelRowJSON(
                name=name.strip(),
                quantity=_validated_nullable_string(entry.get("quantity"), field=f"ores[{index}].quantity"),
                price=_validated_nullable_string(entry.get("price"), field=f"ores[{index}].price"),
            )
        )
    return OrePanelJSON(
        panel_type="ore_panel",
        planet_name=_validated_nullable_string(payload.get("planet_name"), field="planet_name"),
        ores=tuple(rows),
    )


def _validate_planet_panel_payload(payload: object) -> PlanetPanelJSON:
    if not isinstance(payload, dict):
        raise ValueError("planet_panel payload must be an object")
    expected_keys = {"panel_type", "planet_name", "level", "upgrades", "cash"}
    extra_keys = set(payload.keys()) - expected_keys
    if extra_keys:
        raise ValueError(f"planet_panel payload has extra keys: {sorted(extra_keys)}")
    if payload.get("panel_type") != "planet_panel":
        raise ValueError("planet_panel payload has invalid panel_type")
    upgrades = payload.get("upgrades")
    if not isinstance(upgrades, dict):
        raise ValueError("planet_panel payload upgrades must be an object")
    expected_upgrade_keys = {"mining_cost", "speed_cost", "cargo_cost"}
    upgrade_extra = set(upgrades.keys()) - expected_upgrade_keys
    if upgrade_extra:
        raise ValueError(f"planet_panel upgrades has extra keys: {sorted(upgrade_extra)}")
    return PlanetPanelJSON(
        panel_type="planet_panel",
        planet_name=_validated_nullable_string(payload.get("planet_name"), field="planet_name"),
        level=_validated_nullable_string(payload.get("level"), field="level"),
        upgrades=PlanetPanelUpgradesJSON(
            mining_cost=_validated_nullable_string(upgrades.get("mining_cost"), field="upgrades.mining_cost"),
            speed_cost=_validated_nullable_string(upgrades.get("speed_cost"), field="upgrades.speed_cost"),
            cargo_cost=_validated_nullable_string(upgrades.get("cargo_cost"), field="upgrades.cargo_cost"),
        ),
        cash=_validated_nullable_string(payload.get("cash"), field="cash"),
    )


def _semantic_validate_ore_panel_payload(
    payload: OrePanelJSON,
    *,
    known_resource_names: tuple[str, ...],
) -> OrePanelJSON:
    known_lookup = {
        normalize_resource_row_name(name)
        for name in known_resource_names
        if normalize_resource_row_name(name)
    }
    if not payload.ores:
        raise ValueError("invalid_ore_name:no_rows")
    normalized_rows: list[OrePanelRowJSON] = []
    for row in payload.ores:
        reject_reason = resource_row_name_reject_reason(row.name)
        if reject_reason is not None:
            raise ValueError(f"invalid_ore_name:{reject_reason}:{row.name!r}")
        normalized_name = normalize_resource_row_name(row.name)
        if not normalized_name or normalized_name not in known_lookup:
            raise ValueError(f"invalid_ore_name:{row.name!r}")
        if row.name.strip().lower() in {"the", "a", "an"}:
            raise ValueError(f"invalid_ore_name:{row.name!r}")
        if row.quantity is None or not _NULLABLE_COMPACT_RE.fullmatch(row.quantity):
            raise ValueError(f"invalid_quantity:{row.quantity!r}")
        if _parse_compact_number_text(row.quantity) is None:
            raise ValueError(f"invalid_quantity:{row.quantity!r}")
        if row.price is not None:
            if not _NULLABLE_COMPACT_RE.fullmatch(row.price):
                raise ValueError(f"implausible_cost:{row.price!r}")
            if _parse_compact_number_text(row.price) is None:
                raise ValueError(f"implausible_cost:{row.price!r}")
        normalized_rows.append(
            OrePanelRowJSON(
                name=normalized_name,
                quantity=row.quantity,
                price=row.price,
            )
        )
    return OrePanelJSON(
        panel_type=payload.panel_type,
        planet_name=payload.planet_name,
        ores=tuple(normalized_rows),
        backend=payload.backend,
    )


def _semantic_validate_planet_panel_payload(
    payload: PlanetPanelJSON,
    *,
    planet_level_min: int,
    planet_level_max: int,
    upgrade_cost_min: int,
    upgrade_cost_max: int,
) -> PlanetPanelJSON:
    if payload.planet_name is not None and _has_title_prose(payload.planet_name):
        raise ValueError(f"invalid_title_prose:{payload.planet_name!r}")
    if payload.planet_name is not None and not is_plausible_planet_title(payload.planet_name):
        raise ValueError(f"invalid_title_prose:{payload.planet_name!r}")
    if payload.level is not None:
        level_value = _parse_strict_int(payload.level)
        if level_value is None or level_value < planet_level_min or level_value > planet_level_max:
            raise ValueError(f"implausible_level:{payload.level!r}")
    parsed_costs: list[int] = []
    for label, raw_value in (
        ("mining_cost", payload.upgrades.mining_cost),
        ("speed_cost", payload.upgrades.speed_cost),
        ("cargo_cost", payload.upgrades.cargo_cost),
    ):
        if raw_value is None:
            continue
        if not _NULLABLE_COMPACT_RE.fullmatch(raw_value):
            raise ValueError(f"implausible_cost:{label}={raw_value!r}")
        value = _parse_compact_number_text(raw_value)
        if value is None or value < upgrade_cost_min or value > upgrade_cost_max:
            raise ValueError(f"implausible_cost:{label}={raw_value!r}")
        parsed_costs.append(value)
    if payload.planet_name is None and payload.level is None and not parsed_costs and payload.cash is None:
        raise ValueError("invalid_title_prose:empty_panel")
    return payload


def _iter_backends(backend: object):
    if isinstance(backend, HybridPerceptionBackend):
        if backend.primary is not None:
            yield from _iter_backends(backend.primary)
        if backend.fallback is not None:
            yield from _iter_backends(backend.fallback)
        return
    if backend is not None:
        yield backend


def read_text_from_backends(
    backend: object,
    image: Image.Image,
    *,
    prompt: str = "",
    mode: str = "generic",
    allowed_backend_names: tuple[str, ...] = (),
) -> PerceptionRead:
    allowed = {name.lower() for name in allowed_backend_names}
    backend_name = str(getattr(backend, "name", "") or "")
    for candidate in _iter_backends(backend):
        name = str(getattr(candidate, "name", "") or "").lower()
        if allowed and name not in allowed:
            continue
        available = getattr(candidate, "available", None)
        if callable(available) and not available():
            continue
        result = candidate.read_text(image, prompt=prompt, mode=mode)
        if result.value:
            return result
    return PerceptionRead(value="", backend=backend_name, confidence=0.0)


def read_ore_panel_json(backend: object, image: Image.Image) -> OrePanelJSON | None:
    for candidate in _iter_backends(backend):
        name = str(getattr(candidate, "name", "") or "").lower()
        available = getattr(candidate, "available", None)
        if name != "openai" or (callable(available) and not available()):
            continue
        reader = getattr(candidate, "read_ore_panel_json", None)
        if callable(reader):
            return reader(image)
    return None


def read_planet_panel_json(backend: object, image: Image.Image) -> PlanetPanelJSON | None:
    for candidate in _iter_backends(backend):
        name = str(getattr(candidate, "name", "") or "").lower()
        available = getattr(candidate, "available", None)
        if name != "openai" or (callable(available) and not available()):
            continue
        reader = getattr(candidate, "read_planet_panel_json", None)
        if callable(reader):
            return reader(image)
    return None


@dataclass(slots=True)
class LegacyPerceptionBackend:
    name: str = "legacy"
    _rapidocr: object | None = None

    def available(self) -> bool:
        return True

    def _rapidocr_engine(self):
        if self._rapidocr is not None:
            return self._rapidocr or None
        if RapidOCR is None:
            self._rapidocr = False
            return None
        try:
            self._rapidocr = RapidOCR()
        except Exception:
            self._rapidocr = False
            return None
        return self._rapidocr

    def _tesseract_text(self, image: Image.Image, *, mode: str) -> str:
        if pytesseract is None:
            return ""
        psm = 7
        whitelist = ""
        if mode == "numeric":
            whitelist = "0123456789"
        elif mode == "planet_title":
            whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. "
        elif mode == "ore_qty":
            whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ., "
        else:
            psm = 6
        cfg = f"--psm {psm}"
        if whitelist:
            cfg += f" -c tessedit_char_whitelist={whitelist}"
        try:
            return pytesseract.image_to_string(image, config=cfg).strip()
        except Exception:
            return ""

    def _rapidocr_text(self, image: Image.Image) -> str:
        engine = self._rapidocr_engine()
        if engine is None:
            return ""
        try:
            arr = np.array(image)
            result, _ = engine(arr)
        except Exception:
            return ""
        if not result:
            return ""
        texts = []
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
                texts.append(item[1])
        return " ".join(texts).strip()

    def _normalize(self, text: str, *, mode: str) -> str:
        if mode in {"planet_panel", "ore_panel"}:
            cleaned = _normalize_multiline_text(text)
        else:
            cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if mode in {"planet_title", "ore_qty"}:
            return cleaned.upper()
        return cleaned

    def _best_candidate(self, candidates: list[str], *, mode: str) -> str:
        normalized = [self._normalize(text, mode=mode) for text in candidates if str(text or "").strip()]
        if not normalized:
            return ""
        if mode == "numeric":
            normalized.sort(key=lambda value: (sum(ch.isdigit() for ch in value), len(value)), reverse=True)
        else:
            normalized.sort(key=lambda value: (len(value), sum(ch.isalnum() for ch in value)), reverse=True)
        return normalized[0]

    def read_text(self, image: Image.Image, *, prompt: str = "", mode: str = "generic") -> PerceptionRead:
        _ = prompt
        candidates = [
            self._tesseract_text(image, mode=mode),
            self._rapidocr_text(image),
        ]
        text = self._best_candidate(candidates, mode=mode)
        return PerceptionRead(value=text, backend=self.name, confidence=0.85 if text else 0.0)


@dataclass(slots=True)
class OpenAIPerceptionBackend:
    name: str = "openai"
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 96
    enabled: bool = True
    known_ore_names: tuple[str, ...] = ORE_NAMES
    planet_level_min: int = 1
    planet_level_max: int = 999
    upgrade_cost_min: int = 50
    upgrade_cost_max: int = 1_000_000_000_000
    _client: object | None = None

    def available(self) -> bool:
        return bool(self.enabled and OpenAI is not None and os.getenv("OPENAI_API_KEY"))

    def _client_instance(self):
        if self._client is None and self.available():
            self._client = OpenAI()
        return self._client

    def _image_url(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _structured_json_response(
        self,
        image: Image.Image,
        *,
        panel_type: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> object:
        client = self._client_instance()
        if client is None:
            raise StructuredPerceptionError(
                backend=self.name,
                panel_type=panel_type,
                reason="client_unavailable",
            )
        image_url = self._image_url(image)
        try:
            return client.responses.create(
                model=self.model,
                max_output_tokens=self.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise StructuredPerceptionError(
                backend=self.name,
                panel_type=panel_type,
                reason=f"request_failed: {exc}",
            ) from exc

    def _decode_json_payload(
        self,
        response: object,
        *,
        panel_type: str,
        shape_validator,
        semantic_validator,
    ):
        raw_output = _extract_response_text(response)
        if not raw_output:
            raise StructuredPerceptionError(
                backend=self.name,
                panel_type=panel_type,
                reason="empty_output",
                raw_output=raw_output,
            )
        try:
            payload = json.loads(raw_output)
        except Exception as exc:
            print(f"[PERCEPTION] {panel_type} semantic_reject reason=invalid_json raw_output={raw_output!r}")
            raise StructuredPerceptionError(
                backend=self.name,
                panel_type=panel_type,
                reason=f"invalid_json:{exc}",
                raw_output=raw_output,
            ) from exc
        try:
            shaped = shape_validator(payload)
        except Exception as exc:
            print(f"[PERCEPTION] {panel_type} semantic_reject reason=invalid_schema raw_output={raw_output!r}")
            raise StructuredPerceptionError(
                backend=self.name,
                panel_type=panel_type,
                reason=f"invalid_schema:{exc}",
                raw_output=raw_output,
            ) from exc
        try:
            return semantic_validator(shaped)
        except Exception as exc:
            reason = str(exc) or "semantic_reject"
            print(f"[PERCEPTION] {panel_type} semantic_reject reason={reason} raw_output={raw_output!r}")
            raise StructuredPerceptionError(
                backend=self.name,
                panel_type=panel_type,
                reason=reason,
                raw_output=raw_output,
            ) from exc

    def read_text(self, image: Image.Image, *, prompt: str = "", mode: str = "generic") -> PerceptionRead:
        if mode in {"ore_panel", "planet_panel"}:
            return PerceptionRead(value="", backend=self.name, confidence=0.0)
        client = self._client_instance()
        if client is None:
            return PerceptionRead(value="", backend=self.name, confidence=0.0)
        image_url = self._image_url(image)
        try:
            response = client.responses.create(
                model=self.model,
                max_output_tokens=self.max_output_tokens,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt or "Read the visible UI text exactly."},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
            )
        except Exception:
            return PerceptionRead(value="", backend=self.name, confidence=0.0)
        text = _extract_response_text(response)
        if text.upper() == "UNREADABLE":
            text = ""
        return PerceptionRead(value=text, backend=self.name, confidence=0.95 if text else 0.0)

    def read_ore_panel_json(self, image: Image.Image) -> OrePanelJSON:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "panel_type": {"type": "string", "const": "ore_panel"},
                "planet_name": {"type": ["string", "null"]},
                "ores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": ["string", "null"]},
                            "price": {"type": ["string", "null"]},
                        },
                        "required": ["name", "quantity", "price"],
                    },
                },
            },
            "required": ["panel_type", "planet_name", "ores"],
        }
        prompt = (
            "You are parsing a cropped Idle Planet Miner ore panel.\n"
            "Return JSON only.\n"
            "No markdown.\n"
            "No explanations.\n"
            "No sentence wrappers.\n"
            "No extra keys.\n"
            "Use null when unreadable.\n"
            "Each row name must be only the visible resource label.\n"
            "No sentences.\n"
            "No quotes.\n"
            "No units.\n"
            "No percentages.\n"
            "Do not include quantities, prices, level text, speed text, ship text, or explanations.\n"
            "Do not output labels like Ship Speed, Mining Rate, Level, version text, or UI headers.\n"
            "If a row is not a valid resource row, omit it instead of guessing.\n"
            "Each quantity must be the raw visible numeric string only.\n"
            "Each price must be the raw visible numeric string only or null.\n"
            "Schema:\n"
            '{"panel_type":"ore_panel","planet_name":"string or null","ores":[{"name":"string","quantity":"string or null","price":"string or null"}]}'
        )
        response = self._structured_json_response(
            image,
            panel_type="ore_panel",
            prompt=prompt,
            schema_name="ore_panel",
            schema=schema,
        )
        return self._decode_json_payload(
            response,
            panel_type="ore_panel",
            shape_validator=_validate_ore_panel_payload,
            semantic_validator=lambda payload: _semantic_validate_ore_panel_payload(
                payload,
                known_resource_names=RESOURCE_ROW_NAMES,
            ),
        )

    def read_planet_panel_json(self, image: Image.Image) -> PlanetPanelJSON:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "panel_type": {"type": "string", "const": "planet_panel"},
                "planet_name": {"type": ["string", "null"]},
                "level": {"type": ["string", "null"]},
                "upgrades": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mining_cost": {"type": ["string", "null"]},
                        "speed_cost": {"type": ["string", "null"]},
                        "cargo_cost": {"type": ["string", "null"]},
                    },
                    "required": ["mining_cost", "speed_cost", "cargo_cost"],
                },
                "cash": {"type": ["string", "null"]},
            },
            "required": ["panel_type", "planet_name", "level", "upgrades", "cash"],
        }
        prompt = (
            "You are parsing a cropped Idle Planet Miner planet panel.\n"
            "Return JSON only.\n"
            "No markdown.\n"
            "No explanations.\n"
            "No sentence wrappers.\n"
            "No extra keys.\n"
            "Use null when unreadable.\n"
            "Planet title must be the title text only.\n"
            "Level must be the raw visible integer string only or null.\n"
            "Upgrade costs must be the raw visible numeric strings only or null.\n"
            "Cash must be the raw visible numeric string only or null.\n"
            "Schema:\n"
            '{"panel_type":"planet_panel","planet_name":"string or null","level":"string or null","upgrades":{"mining_cost":"string or null","speed_cost":"string or null","cargo_cost":"string or null"},"cash":"string or null"}'
        )
        response = self._structured_json_response(
            image,
            panel_type="planet_panel",
            prompt=prompt,
            schema_name="planet_panel",
            schema=schema,
        )
        return self._decode_json_payload(
            response,
            panel_type="planet_panel",
            shape_validator=_validate_planet_panel_payload,
            semantic_validator=lambda payload: _semantic_validate_planet_panel_payload(
                payload,
                planet_level_min=self.planet_level_min,
                planet_level_max=self.planet_level_max,
                upgrade_cost_min=self.upgrade_cost_min,
                upgrade_cost_max=self.upgrade_cost_max,
            ),
        )


@dataclass(slots=True)
class WindowsOcrPerceptionBackend:
    name: str = "windows"

    def available(self) -> bool:
        return bool(
            os.name == "nt"
            and OcrEngine is not None
            and StorageFile is not None
            and FileAccessMode is not None
            and BitmapDecoder is not None
        )

    async def _read_path(self, path: str) -> str:
        if not self.available():
            return ""
        file = await StorageFile.get_file_from_path_async(path)
        stream = await file.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            return ""
        result = await engine.recognize_async(bitmap)
        return str(getattr(result, "text", "") or "").strip()

    def read_text(self, image: Image.Image, *, prompt: str = "", mode: str = "generic") -> PerceptionRead:
        _ = prompt
        if not self.available():
            return PerceptionRead(value="", backend=self.name, confidence=0.0)
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temp_path = handle.name
            image.save(temp_path, format="PNG")
            text = asyncio.run(self._read_path(temp_path))
        except Exception:
            text = ""
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        if mode in {"planet_panel", "ore_panel"}:
            text = _normalize_multiline_text(text)
        else:
            text = re.sub(r"\s+", " ", text).strip()
        return PerceptionRead(value=text, backend=self.name, confidence=0.9 if text else 0.0)


@dataclass(slots=True)
class HybridPerceptionBackend:
    name: str = "hybrid"
    primary: PerceptionBackend | None = None
    fallback: PerceptionBackend | None = None

    def available(self) -> bool:
        return bool((self.primary and self.primary.available()) or (self.fallback and self.fallback.available()))

    def read_text(self, image: Image.Image, *, prompt: str = "", mode: str = "generic") -> PerceptionRead:
        for backend in (self.primary, self.fallback):
            if backend is None or not backend.available():
                continue
            result = backend.read_text(image, prompt=prompt, mode=mode)
            if result.value:
                return result
        return PerceptionRead(value="", backend=self.name, confidence=0.0)


def create_perception_backend(
    name: str,
    *,
    model: str = "gpt-4.1-mini",
    hybrid_order: str = "windows_first",
    openai_enabled: bool = True,
    openai_max_output_tokens: int = 96,
    known_ore_names: tuple[str, ...] = ORE_NAMES,
    planet_level_min: int = 1,
    planet_level_max: int = 999,
    upgrade_cost_min: int = 50,
    upgrade_cost_max: int = 1_000_000_000_000,
) -> PerceptionBackend:
    normalized = str(name or "hybrid").lower()
    if normalized == "legacy":
        return LegacyPerceptionBackend()
    if normalized in {"windows", "windows_ocr"}:
        return WindowsOcrPerceptionBackend()
    if normalized == "openai":
        return OpenAIPerceptionBackend(
            model=model,
            max_output_tokens=openai_max_output_tokens,
            enabled=openai_enabled,
            known_ore_names=known_ore_names,
            planet_level_min=planet_level_min,
            planet_level_max=planet_level_max,
            upgrade_cost_min=upgrade_cost_min,
            upgrade_cost_max=upgrade_cost_max,
        )

    openai_backend = OpenAIPerceptionBackend(
        model=model,
        max_output_tokens=openai_max_output_tokens,
        enabled=openai_enabled,
        known_ore_names=known_ore_names,
        planet_level_min=planet_level_min,
        planet_level_max=planet_level_max,
        upgrade_cost_min=upgrade_cost_min,
        upgrade_cost_max=upgrade_cost_max,
    )
    legacy_backend = LegacyPerceptionBackend()
    windows_backend = WindowsOcrPerceptionBackend()
    if str(hybrid_order or "openai_first").lower() == "legacy_first":
        return HybridPerceptionBackend(
            primary=legacy_backend,
            fallback=HybridPerceptionBackend(primary=windows_backend, fallback=openai_backend),
        )
    if str(hybrid_order or "windows_first").lower() == "openai_first":
        return HybridPerceptionBackend(
            primary=openai_backend,
            fallback=HybridPerceptionBackend(primary=windows_backend, fallback=legacy_backend),
        )
    return HybridPerceptionBackend(
        primary=windows_backend,
        fallback=HybridPerceptionBackend(primary=openai_backend, fallback=legacy_backend),
    )
