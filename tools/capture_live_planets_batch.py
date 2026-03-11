from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ipm.app import build_application
from ipm.focus import ensure_focus_result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one-shot or batch live PlanetsTask artifacts into JSON files.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of planets task runs to capture.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between runs after writing each artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Base output directory for the batch folder.",
    )
    parser.add_argument(
        "--prefix",
        default="task_observability_batch",
        help="Batch folder prefix.",
    )
    args = parser.parse_args(argv)
    if int(args.runs) < 1:
        parser.error("--runs must be >= 1")
    if float(args.sleep_seconds) < 0.0:
        parser.error("--sleep-seconds must be >= 0")
    return args


def _git_stdout(*git_args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *git_args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _repo_metadata() -> dict[str, Any]:
    head = _git_stdout("rev-parse", "--short", "HEAD")
    status = _git_stdout("status", "--short")
    return {
        "repo_root": str(REPO_ROOT),
        "head": head,
        "git_status_short": status.splitlines() if status else [],
        "git_dirty": bool(status),
    }


def _focus_payload(result) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "reason": result.reason,
        "active_title_before": result.active_title_before,
        "active_title_after": result.active_title_after,
        "activation_status": result.activation_status,
        "activation_error": result.activation_error,
        "target_title": result.target_title,
    }


def _run_summary_payload(run_payload: dict[str, Any]) -> dict[str, Any]:
    task_payload = run_payload.get("task") or {}
    details = task_payload.get("details") or {}
    observability = details.get("observability") or {}
    classification = observability.get("classification") or {}
    return {
        "run_index": run_payload.get("run_index"),
        "error": run_payload.get("error"),
        "focus_ok": bool((run_payload.get("focus") or {}).get("ok")),
        "task_ok": task_payload.get("ok"),
        "duration_seconds": run_payload.get("duration_seconds"),
        "decision": details.get("decision"),
        "executed": details.get("executed"),
        "verified": details.get("verified"),
        "run_kind": classification.get("run_kind"),
        "action_outcome": classification.get("action_outcome"),
    }


def _batch_summary_payload(
    *,
    started_at: str,
    output_dir: Path,
    repo: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    run_kind_counts: Counter[str] = Counter()
    action_outcome_counts: Counter[str] = Counter()
    for run_payload in runs:
        summary = _run_summary_payload(run_payload)
        run_kind = str(summary.get("run_kind") or "unknown")
        action_outcome = str(summary.get("action_outcome") or "unknown")
        run_kind_counts[run_kind] += 1
        action_outcome_counts[action_outcome] += 1
    return {
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "repo": repo,
        "run_count": len(runs),
        "run_kind_counts": dict(run_kind_counts),
        "action_outcome_counts": dict(action_outcome_counts),
        "runs": [_run_summary_payload(run_payload) for run_payload in runs],
    }


def capture_batch(*, runs: int, sleep_seconds: float, output_dir: str, prefix: str) -> Path:
    batch_started_at = datetime.now().isoformat(timespec="seconds")
    repo = _repo_metadata()
    batch_dir = REPO_ROOT / output_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"BATCH_DIR={batch_dir}")

    app = build_application()
    run_payloads: list[dict[str, Any]] = []
    for run_index in range(1, runs + 1):
        run_payload: dict[str, Any] = {
            "run_index": run_index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "repo": repo,
        }
        try:
            focus_result = ensure_focus_result(app.config.focus)
            run_payload["focus"] = _focus_payload(focus_result)
            if not focus_result.ok:
                run_payload["duration_seconds"] = None
                run_payload["task"] = None
                run_payload["error"] = "focus_unavailable"
            else:
                started = time.perf_counter()
                result = app.tasks["planets"].run()
                run_payload["duration_seconds"] = round(time.perf_counter() - started, 3)
                run_payload["task"] = {
                    "ok": bool(result.ok),
                    "details": result.details,
                }
                run_payload["error"] = None
        except Exception as exc:
            run_payload["duration_seconds"] = None
            run_payload["task"] = None
            run_payload["error"] = repr(exc)
            run_payload["traceback"] = traceback.format_exc()
        run_path = batch_dir / f"task_run{run_index}.json"
        run_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        print(f"WROTE={run_path}")
        run_payloads.append(run_payload)
        if run_index < runs and sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    summary = _batch_summary_payload(
        started_at=batch_started_at,
        output_dir=batch_dir,
        repo=repo,
        runs=run_payloads,
    )
    summary_path = batch_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY={summary_path}")
    return batch_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    capture_batch(
        runs=int(args.runs),
        sleep_seconds=float(args.sleep_seconds),
        output_dir=str(args.output_dir),
        prefix=str(args.prefix),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
