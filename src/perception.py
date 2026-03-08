from __future__ import annotations

import base64
import hashlib
import io
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import config
import ocr

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass(frozen=True)
class PerceptionResult:
    text: str
    backend: str
    cached: bool = False


class BasePerceptionBackend:
    name = "base"

    def available(self) -> bool:
        return True

    def read_text(self, bbox, *, mode: str = "generic", prompt: str | None = None) -> str:
        raise NotImplementedError

    def answer_question(self, bbox, *, prompt: str) -> str:
        return self.read_text(bbox, mode="generic", prompt=prompt)


class LegacyPerceptionBackend(BasePerceptionBackend):
    name = "legacy"

    def read_text(self, bbox, *, mode: str = "generic", prompt: str | None = None) -> str:
        text = ocr.ocr_read_text(bbox, mode=mode)
        if text:
            return str(text).strip()
        dbg = ocr.ocr_read_debug(bbox, mode=mode)
        return str(dbg.get("text") or "").strip()


class OpenAIVisionBackend(BasePerceptionBackend):
    name = "openai"

    def __init__(self) -> None:
        self._client = None
        self._cache: OrderedDict[str, str] = OrderedDict()

    def available(self) -> bool:
        if not bool(getattr(config, "PERCEPTION_OPENAI_ENABLED", True)):
            return False
        if OpenAI is None:
            return False
        if not os.getenv("OPENAI_API_KEY"):
            return False
        return True

    def _client_instance(self):
        if self._client is None and self.available():
            self._client = OpenAI()
        return self._client

    def _crop_to_data_url(self, bbox) -> tuple[str, str] | tuple[None, None]:
        img, _meta = ocr.capture_bbox(bbox)
        ok, _reason = ocr.validate_crop(img, bbox, "generic")
        if not ok or img is None:
            return None, None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        image_hash = hashlib.sha1(image_bytes).hexdigest()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}", image_hash

    def _cached(self, key: str) -> str:
        value = self._cache.get(key, "")
        if value:
            self._cache.move_to_end(key)
        return value

    def _remember(self, key: str, value: str) -> None:
        if not key:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        max_entries = max(8, int(getattr(config, "PERCEPTION_CACHE_SIZE", 64)))
        while len(self._cache) > max_entries:
            self._cache.popitem(last=False)

    def read_text(self, bbox, *, mode: str = "generic", prompt: str | None = None) -> str:
        if not self.available():
            return ""
        data_url, image_hash = self._crop_to_data_url(bbox)
        if not data_url or not image_hash:
            return ""

        task_prompt = str(prompt or getattr(config, "PERCEPTION_DEFAULT_TEXT_PROMPT", "")).strip()
        if not task_prompt:
            if mode == "planet_title":
                task_prompt = (
                    "Read the visible UI title exactly as shown. "
                    "Return only the title text, including leading numbers and punctuation. "
                    "Example: 2. DRASTA. If unreadable, return UNREADABLE."
                )
            else:
                task_prompt = (
                    "Read the visible text in this UI crop exactly as shown. "
                    "Return only the text. If unreadable, return UNREADABLE."
                )

        cache_key = hashlib.sha1(f"{mode}\n{task_prompt}\n{image_hash}".encode("utf-8")).hexdigest()
        cached = self._cached(cache_key)
        if cached:
            return cached

        client = self._client_instance()
        if client is None:
            return ""

        response = client.responses.create(
            model=str(getattr(config, "PERCEPTION_OPENAI_MODEL", "gpt-4.1-mini")),
            max_output_tokens=int(getattr(config, "PERCEPTION_OPENAI_MAX_OUTPUT_TOKENS", 64)),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": task_prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text or text.upper() == "UNREADABLE":
            return ""
        self._remember(cache_key, text)
        return text


class HybridPerceptionBackend(BasePerceptionBackend):
    name = "hybrid"

    def __init__(self) -> None:
        self._legacy = LegacyPerceptionBackend()
        self._openai = OpenAIVisionBackend()

    def read_text(self, bbox, *, mode: str = "generic", prompt: str | None = None) -> str:
        preference = str(getattr(config, "PERCEPTION_HYBRID_ORDER", "legacy_first")).lower()
        backends = [self._legacy, self._openai] if preference == "legacy_first" else [self._openai, self._legacy]
        for backend in backends:
            if not backend.available():
                continue
            text = backend.read_text(bbox, mode=mode, prompt=prompt)
            if text:
                return text
        return ""

    def answer_question(self, bbox, *, prompt: str) -> str:
        if self._openai.available():
            answer = self._openai.answer_question(bbox, prompt=prompt)
            if answer:
                return answer
        return self._legacy.answer_question(bbox, prompt=prompt)


_BACKEND = None


def get_backend() -> BasePerceptionBackend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    backend_name = str(getattr(config, "PERCEPTION_BACKEND", "hybrid")).lower()
    if backend_name == "openai":
        _BACKEND = OpenAIVisionBackend()
    elif backend_name == "legacy":
        _BACKEND = LegacyPerceptionBackend()
    else:
        _BACKEND = HybridPerceptionBackend()
    return _BACKEND


def reset_backend_cache() -> None:
    global _BACKEND
    _BACKEND = None


def read_text(bbox, *, mode: str = "generic", prompt: str | None = None) -> PerceptionResult:
    backend = get_backend()
    text = backend.read_text(bbox, mode=mode, prompt=prompt)
    return PerceptionResult(text=text, backend=getattr(backend, "name", "unknown"))


def read_planet_title_text(bbox) -> PerceptionResult:
    prompt = str(
        getattr(
            config,
            "PERCEPTION_PLANET_TITLE_PROMPT",
            (
                "Read the planet title exactly as displayed in this game UI crop. "
                "Return only the visible title text, preserving the leading number and punctuation. "
                "Example: 2. DRASTA. If unreadable, return UNREADABLE."
            ),
        )
    )
    return read_text(bbox, mode="planet_title", prompt=prompt)


def read_number(
    bbox,
    *,
    mode: str = "generic",
    prompt: str | None = None,
) -> PerceptionResult:
    if not prompt:
        if mode == "ore_qty":
            prompt = (
                "Read the ore quantity shown in this UI crop. "
                "Return only the quantity text exactly, including suffixes like K, M, B if present. "
                "If unreadable, return UNREADABLE."
            )
        elif mode == "hud_cash":
            prompt = (
                "Read the price or cash amount shown in this UI crop. "
                "Return only the amount text exactly as displayed. "
                "If unreadable, return UNREADABLE."
            )
        else:
            prompt = (
                "Read the numeric value shown in this UI crop. "
                "Return only the visible value text. If unreadable, return UNREADABLE."
            )
    return read_text(bbox, mode=mode, prompt=prompt)


def read_number_value(
    bbox,
    *,
    mode: str = "generic",
    prompt: str | None = None,
) -> tuple[Optional[int], PerceptionResult]:
    result = read_number(bbox, mode=mode, prompt=prompt)
    value = ocr.parse_compact_number_for_mode(result.text, mode=mode)
    return value, result


def answer_question(bbox, *, prompt: str) -> PerceptionResult:
    backend = get_backend()
    text = backend.answer_question(bbox, prompt=prompt)
    return PerceptionResult(text=text, backend=getattr(backend, "name", "unknown"))
