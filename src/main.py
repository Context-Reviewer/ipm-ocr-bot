from __future__ import annotations

import argparse

from ipm.app import build_application


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starfield-probe-once", action="store_true")
    parser.add_argument("--discover-nearest-planet-once", action="store_true")
    args = parser.parse_args(argv)
    app = build_application()
    if args.discover_nearest_planet_once:
        return app.run_discover_nearest_planet_once()
    if args.starfield_probe_once:
        return app.run_starfield_probe_once()
    app.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
