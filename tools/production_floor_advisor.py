from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ipm.app import build_application
from ipm.readers.inventory_panel import InventoryPanelReader
from production_floor_advisor import compute_production_floor_advice_from_mapping
from production_floor_live_state import ProductionFloorLiveStateReader


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute read-only production protection floors from explicit assignment and inventory state.",
    )
    parser.add_argument(
        "--state-json",
        required=True,
        help="Path to a JSON file containing ores/bars/items inventory plus smelter_queue/crafter_queue assignments.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--overlay-live-bars-items",
        action="store_true",
        help="Override bars/items inventory in the provided state JSON using the live Resources panel.",
    )
    return parser.parse_args(argv)


def _overlay_live_bars_items(payload: dict) -> tuple[dict, dict]:
    app = build_application()
    live_reader = ProductionFloorLiveStateReader(
        config=app.config,
        rects=app.rects,
        capture=app.capture_backend,
        actions=app.actions,
        inventory_reader=InventoryPanelReader(app.config, app.rects, app.capture_backend, app.perception_backend),
        perception=app.perception_backend,
    )
    live_state = live_reader.read()
    merged = dict(payload)
    merged["bars"] = dict(live_state.get("bars") or {})
    merged["items"] = dict(live_state.get("items") or {})
    if live_state.get("smelter_queue"):
        merged["smelter_queue"] = dict(live_state.get("smelter_queue") or {})
    if live_state.get("crafter_queue"):
        merged["crafter_queue"] = dict(live_state.get("crafter_queue") or {})
    return merged, live_state


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    state_path = Path(str(args.state_json))
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    live_state = None
    if bool(args.overlay_live_bars_items):
        payload, live_state = _overlay_live_bars_items(payload)
    advice = compute_production_floor_advice_from_mapping(payload)
    if live_state is not None:
        advice["input_mode"] = {
            "active_assignments": (
                "manual_state_snapshot_plus_live_production_overview_overlay"
                if live_state.get("smelter_queue") or live_state.get("crafter_queue")
                else "manual_state_snapshot"
            ),
            "inventory": "manual_state_snapshot_plus_live_resources_panel_overlay",
        }
        advice["live_reader_support"] = {
            "active_assignments": bool(
                live_state.get("seam_status", {})
                .get("active_smelter_assignments", {})
                .get("feasible")
            )
            and bool(
                live_state.get("seam_status", {})
                .get("active_crafter_assignments", {})
                .get("feasible")
            ),
            "ores": True,
            "bars": True,
            "items": True,
        }
        advice["limitations"]["active_assignment_reader_available"] = advice["live_reader_support"]["active_assignments"]
        advice["limitations"]["current_inventory_live_readers"] = {
            "ores": True,
            "bars": True,
            "items": True,
        }
        advice["live_seam_status"] = live_state.get("seam_status")
    rendered = json.dumps(advice, indent=2)
    if args.output_json:
        output_path = Path(str(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"WROTE={output_path}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
