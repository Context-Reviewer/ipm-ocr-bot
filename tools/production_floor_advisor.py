from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from production_floor_advisor import compute_production_floor_advice_from_mapping


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    state_path = Path(str(args.state_json))
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    advice = compute_production_floor_advice_from_mapping(payload)
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
