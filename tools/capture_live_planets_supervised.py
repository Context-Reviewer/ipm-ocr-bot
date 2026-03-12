from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from capture_live_planets_batch import REPO_ROOT, _focus_payload, _repo_metadata, _run_summary_payload

sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ipm.app import build_application
from ipm.focus import ensure_focus_result

_RECENT_CYCLE_TAIL_LIMIT = 5


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supervised repeated PlanetsTask captures with durable session artifacts.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=60.0,
        help="Sleep between cycles after writing artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Base output directory for the session folder.",
    )
    parser.add_argument(
        "--prefix",
        default="task_supervised_planets",
        help="Session folder prefix.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="Maximum cycles to execute. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-consecutive-focus-unavailable",
        type=int,
        default=3,
        help="Stop after this many consecutive focus_unavailable cycles.",
    )
    parser.add_argument(
        "--max-consecutive-task-exceptions",
        type=int,
        default=3,
        help="Stop after this many consecutive task exception cycles.",
    )
    parser.add_argument(
        "--max-consecutive-artifact-write-failures",
        type=int,
        default=3,
        help="Stop after this many consecutive cycles with artifact write failures.",
    )
    args = parser.parse_args(argv)
    if float(args.sleep_seconds) < 0.0:
        parser.error("--sleep-seconds must be >= 0")
    if int(args.max_runs) < 0:
        parser.error("--max-runs must be >= 0")
    if int(args.max_consecutive_focus_unavailable) < 1:
        parser.error("--max-consecutive-focus-unavailable must be >= 1")
    if int(args.max_consecutive_task_exceptions) < 1:
        parser.error("--max-consecutive-task-exceptions must be >= 1")
    if int(args.max_consecutive_artifact_write_failures) < 1:
        parser.error("--max-consecutive-artifact-write-failures must be >= 1")
    return args


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _session_run_summary_payload(run_payload: dict[str, Any]) -> dict[str, Any]:
    summary = _run_summary_payload(run_payload)
    summary["timestamp"] = run_payload.get("timestamp")
    summary["artifact_write_ok"] = bool(run_payload.get("artifact_write_ok", True))
    summary["artifact_path"] = run_payload.get("artifact_path")
    return summary


def _session_summary_payload(
    *,
    started_at: str,
    output_dir: Path,
    repo: dict[str, Any],
    runs: list[dict[str, Any]],
    stop_reason: str,
    limits: dict[str, Any],
) -> dict[str, Any]:
    no_op_count = 0
    action_bearing_count = 0
    verified_action_count = 0
    exception_count = 0
    focus_unavailable_count = 0
    artifact_write_failure_count = 0
    run_kind_counts: Counter[str] = Counter()
    action_outcome_counts: Counter[str] = Counter()
    panel_class_counts: Counter[str] = Counter()
    escalation_reason_counts: Counter[str] = Counter()
    slow_panel_class_counts: Counter[str] = Counter()
    last_successful_cycle_at: str | None = None
    last_action_bearing_cycle_at: str | None = None
    last_exception_at: str | None = None
    summaries: list[dict[str, Any]] = []
    for run_payload in runs:
        summary = _session_run_summary_payload(run_payload)
        summaries.append(summary)
        run_kind = str(summary.get("run_kind") or "unknown")
        action_outcome = str(summary.get("action_outcome") or "unknown")
        run_kind_counts[run_kind] += 1
        action_outcome_counts[action_outcome] += 1
        if run_kind == "no_op":
            no_op_count += 1
        if run_kind == "action_bearing":
            action_bearing_count += 1
            last_action_bearing_cycle_at = str(run_payload.get("timestamp") or "")
        if action_outcome == "verified":
            verified_action_count += 1
        if run_payload.get("error") == "focus_unavailable":
            focus_unavailable_count += 1
        if bool(run_payload.get("task_exception")):
            exception_count += 1
            last_exception_at = str(run_payload.get("timestamp") or "")
        if not bool(run_payload.get("artifact_write_ok", True)):
            artifact_write_failure_count += 1
        if not run_payload.get("error") and not bool(run_payload.get("task_exception")):
            last_successful_cycle_at = str(run_payload.get("timestamp") or "")
        observability = ((run_payload.get("task") or {}).get("details") or {}).get("observability") or {}
        scan = observability.get("scan") or {}
        for key, counter in (
            ("panel_class_counts", panel_class_counts),
            ("escalation_reason_counts", escalation_reason_counts),
            ("slow_panel_class_counts", slow_panel_class_counts),
        ):
            values = scan.get(key) or {}
            for name, count in values.items():
                counter[str(name)] += int(count)
    return {
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "repo": repo,
        "output_dir": str(output_dir),
        "total_cycles": len(runs),
        "no_op_count": no_op_count,
        "action_bearing_count": action_bearing_count,
        "verified_action_count": verified_action_count,
        "exception_count": exception_count,
        "focus_unavailable_count": focus_unavailable_count,
        "artifact_write_failure_count": artifact_write_failure_count,
        "run_kind_counts": dict(run_kind_counts),
        "action_outcome_counts": dict(action_outcome_counts),
        "panel_class_counts": dict(panel_class_counts),
        "escalation_reason_counts": dict(escalation_reason_counts),
        "slow_panel_class_counts": dict(slow_panel_class_counts),
        "last_successful_cycle_at": last_successful_cycle_at,
        "last_action_bearing_cycle_at": last_action_bearing_cycle_at,
        "last_exception_at": last_exception_at,
        "stop_reason": stop_reason,
        "limits": limits,
        "recent_cycles_tail": summaries[-_RECENT_CYCLE_TAIL_LIMIT:],
        "runs": summaries,
    }


def run_supervised_session(
    *,
    sleep_seconds: float,
    output_dir: str,
    prefix: str,
    max_runs: int,
    max_consecutive_focus_unavailable: int,
    max_consecutive_task_exceptions: int,
    max_consecutive_artifact_write_failures: int,
) -> Path:
    session_started_at = datetime.now().isoformat(timespec="seconds")
    repo = _repo_metadata()
    session_dir = REPO_ROOT / output_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    summary_path = session_dir / "session_summary.json"
    print(f"SESSION_DIR={session_dir}")

    limits = {
        "max_runs": max_runs,
        "sleep_seconds": sleep_seconds,
        "max_consecutive_focus_unavailable": max_consecutive_focus_unavailable,
        "max_consecutive_task_exceptions": max_consecutive_task_exceptions,
        "max_consecutive_artifact_write_failures": max_consecutive_artifact_write_failures,
    }
    run_payloads: list[dict[str, Any]] = []
    consecutive_focus_unavailable = 0
    consecutive_task_exceptions = 0
    consecutive_artifact_write_failures = 0
    stop_reason = "running"

    app = build_application()
    cycle_index = 0
    while True:
        cycle_index += 1
        run_payload: dict[str, Any] = {
            "run_index": cycle_index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "repo": repo,
            "artifact_write_ok": True,
            "task_exception": False,
        }
        cycle_write_failed = False
        try:
            focus_result = ensure_focus_result(app.config.focus)
            run_payload["focus"] = _focus_payload(focus_result)
            if not focus_result.ok:
                run_payload["duration_seconds"] = None
                run_payload["task"] = None
                run_payload["error"] = "focus_unavailable"
                consecutive_focus_unavailable += 1
                consecutive_task_exceptions = 0
            else:
                consecutive_focus_unavailable = 0
                started = time.perf_counter()
                result = app.tasks["planets"].run()
                run_payload["duration_seconds"] = round(time.perf_counter() - started, 3)
                run_payload["task"] = {
                    "ok": bool(result.ok),
                    "details": result.details,
                }
                run_payload["error"] = None
                consecutive_task_exceptions = 0
        except Exception as exc:
            run_payload["duration_seconds"] = None
            run_payload["task"] = None
            run_payload["error"] = repr(exc)
            run_payload["traceback"] = traceback.format_exc()
            run_payload["task_exception"] = True
            consecutive_task_exceptions += 1
            consecutive_focus_unavailable = 0

        run_path = session_dir / f"task_run{cycle_index}.json"
        try:
            _write_json_atomic(run_path, run_payload)
            run_payload["artifact_path"] = str(run_path)
        except Exception as exc:
            cycle_write_failed = True
            run_payload["artifact_write_ok"] = False
            run_payload["artifact_write_error"] = repr(exc)
            run_payload["artifact_write_traceback"] = traceback.format_exc()
            print(f"ARTIFACT_WRITE_FAILED path={run_path} error={exc!r}", file=sys.stderr)

        run_payloads.append(run_payload)
        if cycle_write_failed:
            consecutive_artifact_write_failures += 1
        else:
            consecutive_artifact_write_failures = 0

        if max_runs and cycle_index >= max_runs:
            stop_reason = "max_runs_reached"
        elif consecutive_focus_unavailable >= max_consecutive_focus_unavailable:
            stop_reason = "repeated_focus_unavailable"
        elif consecutive_task_exceptions >= max_consecutive_task_exceptions:
            stop_reason = "repeated_task_exceptions"
        elif consecutive_artifact_write_failures >= max_consecutive_artifact_write_failures:
            stop_reason = "repeated_artifact_write_failures"

        summary = _session_summary_payload(
            started_at=session_started_at,
            output_dir=session_dir,
            repo=repo,
            runs=run_payloads,
            stop_reason=stop_reason,
            limits=limits,
        )
        try:
            _write_json_atomic(summary_path, summary)
        except Exception as exc:
            print(f"SESSION_SUMMARY_WRITE_FAILED path={summary_path} error={exc!r}", file=sys.stderr)
            raise

        print(f"WROTE={run_path}")
        print(f"SUMMARY={summary_path}")
        if stop_reason != "running":
            break
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    return session_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_supervised_session(
        sleep_seconds=float(args.sleep_seconds),
        output_dir=str(args.output_dir),
        prefix=str(args.prefix),
        max_runs=int(args.max_runs),
        max_consecutive_focus_unavailable=int(args.max_consecutive_focus_unavailable),
        max_consecutive_task_exceptions=int(args.max_consecutive_task_exceptions),
        max_consecutive_artifact_write_failures=int(args.max_consecutive_artifact_write_failures),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
