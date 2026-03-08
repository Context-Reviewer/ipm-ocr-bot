from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


Rect = tuple[int, int, int, int]


@dataclass(slots=True)
class RectStore:
    path: Path | None
    rects: dict[str, Rect]

    @classmethod
    def load(cls, path: str | Path) -> "RectStore":
        p = Path(path)
        if not p.exists():
            return cls(path=p, rects={})
        data = json.loads(p.read_text(encoding="utf-8"))
        rects: dict[str, Rect] = {}
        for key, value in (data or {}).items():
            if isinstance(value, (list, tuple)) and len(value) == 4:
                rects[str(key)] = tuple(int(v) for v in value)
        return cls(path=p, rects=rects)

    def get(self, key: str) -> Rect | None:
        rect = self.rects.get(str(key))
        if rect is None or len(rect) != 4:
            return None
        return tuple(int(v) for v in rect)
