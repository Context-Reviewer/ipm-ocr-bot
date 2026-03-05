from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


Rect = Tuple[int, int, int, int]


@dataclass
class RectStore:
    path: Path
    rects: Dict[str, Rect]

    @staticmethod
    def load(path: str | Path) -> "RectStore":
        p = Path(path)
        if not p.exists():
            return RectStore(path=p, rects={})

        data = json.loads(p.read_text(encoding="utf-8"))
        out: Dict[str, Rect] = {}
        for k, v in (data or {}).items():
            if isinstance(v, (list, tuple)) and len(v) == 4:
                x, y, w, h = [int(round(float(n))) for n in v]
                out[str(k)] = (x, y, w, h)
        return RectStore(path=p, rects=out)

    def save(self) -> None:
        # Stable ordering + stable formatting
        payload = {k: list(self.rects[k]) for k in sorted(self.rects.keys())}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)
