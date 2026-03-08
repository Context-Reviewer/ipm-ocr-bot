from __future__ import annotations

import argparse

from ipm.app import build_application


def _positive_rank(value: str) -> int:
    rank = int(value)
    if rank < 1:
        raise argparse.ArgumentTypeError("rank must be >= 1")
    return rank


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starfield-probe-once", action="store_true")
    discovery_group = parser.add_mutually_exclusive_group()
    discovery_group.add_argument("--discover-nearest-planet-once", action="store_true")
    discovery_group.add_argument("--discover-planet-rank", type=_positive_rank, metavar="N")
    args = parser.parse_args(argv)
    app = build_application()
    if args.discover_planet_rank is not None:
        return app.run_discover_planet_rank_once(args.discover_planet_rank)
    if args.discover_nearest_planet_once:
        return app.run_discover_nearest_planet_once()
    if args.starfield_probe_once:
        return app.run_starfield_probe_once()
    app.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
