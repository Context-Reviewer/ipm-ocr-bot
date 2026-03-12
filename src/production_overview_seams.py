from __future__ import annotations

from collections import Counter
from typing import Iterable

import bars_data
import items_data
from ipm.state import ProductionOverviewCardState

_READER_CONTRACT = (
    "read four production overview slots as ProductionOverviewCardState(slot_index=<int>, tab=<smelt|craft>, "
    "output_name=<canonical output>, active=<bool>, timer_text=<optional>, backend=<matcher>)"
)
_PARSER_CONTRACT = "aggregate only cards with output_name and active=true into dict[output_name, active_count]"


def required_production_overview_rects() -> list[str]:
    return [
        "PRODUCTION_CARD1",
        "PRODUCTION_CARD2",
        "PRODUCTION_CARD3",
        "PRODUCTION_CARD4",
    ]


def allowed_overview_outputs(kind: str) -> set[str]:
    normalized = str(kind or "").lower()
    if normalized == "smelt":
        return set(bars_data.list_bars())
    if normalized == "craft":
        return set(items_data.list_items())
    raise ValueError(f"unsupported_assignment_kind:{kind}")


def parse_active_overview_cards(
    cards: Iterable[ProductionOverviewCardState],
    *,
    allowed_outputs: set[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for card in cards:
        output_name = str(getattr(card, "output_name", "") or "").strip()
        if not output_name:
            continue
        if output_name not in allowed_outputs:
            raise ValueError(f"unsupported_assignment_card:{output_name}")
        if bool(getattr(card, "active", False)):
            counts[output_name] += 1
    return dict(counts)


def seam_contract_summary() -> dict[str, object]:
    return {
        "reader_contract": _READER_CONTRACT,
        "parser_contract": _PARSER_CONTRACT,
        "required_rects": required_production_overview_rects(),
    }
