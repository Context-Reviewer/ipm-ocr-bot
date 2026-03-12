from __future__ import annotations

import argparse
from collections import Counter
import json
import signal
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
_WATCHDOG_STATE_KEYS = (
    "dominant_panel_class",
    "dominant_escalation_reason",
    "slow_no_op",
)


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
    parser.add_argument(
        "--max-consecutive-dominant-panel-class",
        type=int,
        default=6,
        help="Stop after this many consecutive cycles with the same dominant non-seed panel_class. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-consecutive-dominant-escalation-reason",
        type=int,
        default=6,
        help="Stop after this many consecutive cycles with the same dominant escalation_reason. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-consecutive-slow-no-op",
        type=int,
        default=0,
        help="Stop after this many consecutive slow no-op cycles. Use 0 to disable.",
    )
    parser.add_argument(
        "--slow-no-op-threshold-seconds",
        type=float,
        default=20.0,
        help="Duration threshold used by the slow no-op watchdog.",
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
    for value, name in (
        (int(args.max_consecutive_dominant_panel_class), "--max-consecutive-dominant-panel-class"),
        (int(args.max_consecutive_dominant_escalation_reason), "--max-consecutive-dominant-escalation-reason"),
        (int(args.max_consecutive_slow_no_op), "--max-consecutive-slow-no-op"),
    ):
        if value < 0:
            parser.error(f"{name} must be >= 0")
        if value == 1:
            parser.error(f"{name} must be 0 or >= 2")
    if float(args.slow_no_op_threshold_seconds) < 0.0:
        parser.error("--slow-no-op-threshold-seconds must be >= 0")
    return args


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt()


def _session_run_summary_payload(run_payload: dict[str, Any]) -> dict[str, Any]:
    summary = _run_summary_payload(run_payload)
    summary["timestamp"] = run_payload.get("timestamp")
    summary["artifact_write_ok"] = bool(run_payload.get("artifact_write_ok", True))
    summary["artifact_path"] = run_payload.get("artifact_path")
    return summary


def _empty_watchdog_pattern_state() -> dict[str, Any]:
    return {
        "pattern": None,
        "consecutive_cycles": 0,
        "run_indexes": [],
        "timestamps": [],
        "metric_values": [],
    }


def _empty_watchdog_state() -> dict[str, Any]:
    return {key: _empty_watchdog_pattern_state() for key in _WATCHDOG_STATE_KEYS}


def _dominant_count_entry(values: dict[str, Any] | None, *, excluded_keys: set[str] | None = None) -> tuple[str | None, int]:
    counts: dict[str, int] = {}
    excluded = excluded_keys or set()
    for name, raw_count in (values or {}).items():
        key = str(name)
        if key in excluded:
            continue
        count = int(raw_count)
        if count > 0:
            counts[key] = count
    if not counts:
        return None, 0
    highest = max(counts.values())
    dominant = [name for name, count in counts.items() if count == highest]
    if len(dominant) != 1:
        return None, 0
    return dominant[0], highest


def _update_watchdog_pattern_state(
    state: dict[str, Any],
    *,
    pattern: str | None,
    run_index: int,
    timestamp: str,
    metric_value: int | float | None,
) -> dict[str, Any]:
    if not pattern:
        return _empty_watchdog_pattern_state()
    if state.get("pattern") == pattern:
        return {
            "pattern": pattern,
            "consecutive_cycles": int(state.get("consecutive_cycles", 0)) + 1,
            "run_indexes": [*list(state.get("run_indexes") or []), run_index],
            "timestamps": [*list(state.get("timestamps") or []), timestamp],
            "metric_values": [*list(state.get("metric_values") or []), metric_value],
        }
    return {
        "pattern": pattern,
        "consecutive_cycles": 1,
        "run_indexes": [run_index],
        "timestamps": [timestamp],
        "metric_values": [metric_value],
    }


def _watchdog_observations(run_payload: dict[str, Any], *, slow_no_op_threshold_seconds: float) -> dict[str, Any]:
    summary = _session_run_summary_payload(run_payload)
    observability = ((run_payload.get("task") or {}).get("details") or {}).get("observability") or {}
    scan = observability.get("scan") or {}
    dominant_panel_class, dominant_panel_class_count = _dominant_count_entry(
        scan.get("panel_class_counts"),
        excluded_keys={"seed_only"},
    )
    dominant_escalation_reason, dominant_escalation_reason_count = _dominant_count_entry(
        scan.get("escalation_reason_counts")
    )
    duration_seconds = summary.get("duration_seconds")
    is_slow_no_op = (
        summary.get("run_kind") == "no_op"
        and isinstance(duration_seconds, (int, float))
        and float(duration_seconds) >= float(slow_no_op_threshold_seconds)
    )
    return {
        "dominant_panel_class": dominant_panel_class,
        "dominant_panel_class_count": dominant_panel_class_count,
        "dominant_escalation_reason": dominant_escalation_reason,
        "dominant_escalation_reason_count": dominant_escalation_reason_count,
        "slow_no_op": bool(is_slow_no_op),
        "slow_no_op_duration_seconds": float(duration_seconds) if is_slow_no_op else None,
    }


def _update_watchdog_state(
    state: dict[str, Any],
    *,
    run_payload: dict[str, Any],
    slow_no_op_threshold_seconds: float,
) -> dict[str, Any]:
    next_state = {
        key: {
            "pattern": value.get("pattern"),
            "consecutive_cycles": int(value.get("consecutive_cycles", 0)),
            "run_indexes": list(value.get("run_indexes") or []),
            "timestamps": list(value.get("timestamps") or []),
            "metric_values": list(value.get("metric_values") or []),
        }
        for key, value in state.items()
    }
    observations = _watchdog_observations(
        run_payload,
        slow_no_op_threshold_seconds=slow_no_op_threshold_seconds,
    )
    run_index = int(run_payload.get("run_index") or 0)
    timestamp = str(run_payload.get("timestamp") or "")
    next_state["dominant_panel_class"] = _update_watchdog_pattern_state(
        next_state["dominant_panel_class"],
        pattern=observations.get("dominant_panel_class"),
        run_index=run_index,
        timestamp=timestamp,
        metric_value=observations.get("dominant_panel_class_count"),
    )
    next_state["dominant_escalation_reason"] = _update_watchdog_pattern_state(
        next_state["dominant_escalation_reason"],
        pattern=observations.get("dominant_escalation_reason"),
        run_index=run_index,
        timestamp=timestamp,
        metric_value=observations.get("dominant_escalation_reason_count"),
    )
    next_state["slow_no_op"] = _update_watchdog_pattern_state(
        next_state["slow_no_op"],
        pattern="slow_no_op" if observations.get("slow_no_op") else None,
        run_index=run_index,
        timestamp=timestamp,
        metric_value=observations.get("slow_no_op_duration_seconds"),
    )
    return next_state


def _watchdog_trigger_payload(
    *,
    kind: str,
    stop_reason: str,
    threshold: int,
    state: dict[str, Any],
    slow_no_op_threshold_seconds: float | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "stop_reason": stop_reason,
        "pattern": state.get("pattern"),
        "consecutive_cycles": int(state.get("consecutive_cycles", 0)),
        "threshold": threshold,
        "run_indexes": list(state.get("run_indexes") or []),
        "timestamps": list(state.get("timestamps") or []),
        "metric_values": list(state.get("metric_values") or []),
    }
    if slow_no_op_threshold_seconds is not None:
        payload["slow_no_op_threshold_seconds"] = slow_no_op_threshold_seconds
    return payload


def _evaluate_watchdog_stop(
    *,
    watchdog_state: dict[str, Any],
    max_consecutive_dominant_panel_class: int,
    max_consecutive_dominant_escalation_reason: int,
    max_consecutive_slow_no_op: int,
    slow_no_op_threshold_seconds: float,
) -> tuple[str | None, dict[str, Any] | None]:
    panel_state = watchdog_state["dominant_panel_class"]
    if max_consecutive_dominant_panel_class and int(panel_state.get("consecutive_cycles", 0)) >= max_consecutive_dominant_panel_class:
        return (
            "repeated_dominant_panel_class",
            _watchdog_trigger_payload(
                kind="dominant_panel_class",
                stop_reason="repeated_dominant_panel_class",
                threshold=max_consecutive_dominant_panel_class,
                state=panel_state,
            ),
        )
    escalation_state = watchdog_state["dominant_escalation_reason"]
    if max_consecutive_dominant_escalation_reason and int(escalation_state.get("consecutive_cycles", 0)) >= max_consecutive_dominant_escalation_reason:
        return (
            "repeated_dominant_escalation_reason",
            _watchdog_trigger_payload(
                kind="dominant_escalation_reason",
                stop_reason="repeated_dominant_escalation_reason",
                threshold=max_consecutive_dominant_escalation_reason,
                state=escalation_state,
            ),
        )
    slow_state = watchdog_state["slow_no_op"]
    if max_consecutive_slow_no_op and int(slow_state.get("consecutive_cycles", 0)) >= max_consecutive_slow_no_op:
        return (
            "repeated_slow_no_op_cycles",
            _watchdog_trigger_payload(
                kind="slow_no_op",
                stop_reason="repeated_slow_no_op_cycles",
                threshold=max_consecutive_slow_no_op,
                state=slow_state,
                slow_no_op_threshold_seconds=slow_no_op_threshold_seconds,
            ),
        )
    return None, None


def _session_summary_payload(
    *,
    started_at: str,
    output_dir: Path,
    repo: dict[str, Any],
    runs: list[dict[str, Any]],
    stop_reason: str,
    limits: dict[str, Any],
    watchdog_state: dict[str, Any] | None = None,
    watchdog_trigger: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
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
        "stop_details": stop_details,
        "limits": limits,
        "watchdog_state": watchdog_state or _empty_watchdog_state(),
        "watchdog_trigger": watchdog_trigger,
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
    max_consecutive_dominant_panel_class: int,
    max_consecutive_dominant_escalation_reason: int,
    max_consecutive_slow_no_op: int,
    slow_no_op_threshold_seconds: float,
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
        "max_consecutive_dominant_panel_class": max_consecutive_dominant_panel_class,
        "max_consecutive_dominant_escalation_reason": max_consecutive_dominant_escalation_reason,
        "max_consecutive_slow_no_op": max_consecutive_slow_no_op,
        "slow_no_op_threshold_seconds": slow_no_op_threshold_seconds,
    }
    run_payloads: list[dict[str, Any]] = []
    consecutive_focus_unavailable = 0
    consecutive_task_exceptions = 0
    consecutive_artifact_write_failures = 0
    watchdog_state = _empty_watchdog_state()
    watchdog_trigger: dict[str, Any] | None = None
    stop_reason = "running"
    stop_details: dict[str, Any] | None = None
    cycle_phase = "startup"
    last_completed_run_index = 0
    previous_sigbreak_handler = None

    app = build_application()
    cycle_index = 0
    if hasattr(signal, "SIGBREAK"):
        previous_sigbreak_handler = signal.getsignal(signal.SIGBREAK)
        signal.signal(signal.SIGBREAK, _raise_keyboard_interrupt)
    try:
        while True:
            cycle_index += 1
            cycle_phase = "active_cycle"
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

            watchdog_state = _update_watchdog_state(
                watchdog_state,
                run_payload=run_payload,
                slow_no_op_threshold_seconds=slow_no_op_threshold_seconds,
            )

            if max_runs and cycle_index >= max_runs:
                stop_reason = "max_runs_reached"
            elif consecutive_focus_unavailable >= max_consecutive_focus_unavailable:
                stop_reason = "repeated_focus_unavailable"
            elif consecutive_task_exceptions >= max_consecutive_task_exceptions:
                stop_reason = "repeated_task_exceptions"
            elif consecutive_artifact_write_failures >= max_consecutive_artifact_write_failures:
                stop_reason = "repeated_artifact_write_failures"
            else:
                stop_reason, watchdog_trigger = _evaluate_watchdog_stop(
                    watchdog_state=watchdog_state,
                    max_consecutive_dominant_panel_class=max_consecutive_dominant_panel_class,
                    max_consecutive_dominant_escalation_reason=max_consecutive_dominant_escalation_reason,
                    max_consecutive_slow_no_op=max_consecutive_slow_no_op,
                    slow_no_op_threshold_seconds=slow_no_op_threshold_seconds,
                )
                stop_reason = stop_reason or "running"

            summary = _session_summary_payload(
                started_at=session_started_at,
                output_dir=session_dir,
                repo=repo,
                runs=run_payloads,
                stop_reason=stop_reason,
                limits=limits,
                watchdog_state=watchdog_state,
                watchdog_trigger=watchdog_trigger,
                stop_details=stop_details,
            )
            try:
                _write_json_atomic(summary_path, summary)
            except Exception as exc:
                print(f"SESSION_SUMMARY_WRITE_FAILED path={summary_path} error={exc!r}", file=sys.stderr)
                raise

            last_completed_run_index = cycle_index
            print(f"WROTE={run_path}")
            print(f"SUMMARY={summary_path}")
            if stop_reason != "running":
                break
            if sleep_seconds > 0.0:
                cycle_phase = "sleep"
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        stop_reason = "operator_interrupt"
        stop_details = {
            "kind": "operator_interrupt",
            "phase": "sleep" if cycle_phase == "sleep" else "active_cycle",
            "last_completed_run_index": last_completed_run_index,
        }
        print(
            f"STOP reason=operator_interrupt phase={stop_details['phase']} "
            f"last_completed_run_index={last_completed_run_index}"
        )
    if stop_reason == "operator_interrupt":
        summary = _session_summary_payload(
            started_at=session_started_at,
            output_dir=session_dir,
            repo=repo,
            runs=run_payloads,
            stop_reason=stop_reason,
            limits=limits,
            watchdog_state=watchdog_state,
            watchdog_trigger=watchdog_trigger,
            stop_details=stop_details,
        )
        try:
            _write_json_atomic(summary_path, summary)
        except Exception as exc:
            print(f"SESSION_SUMMARY_WRITE_FAILED path={summary_path} error={exc!r}", file=sys.stderr)
            raise
        print(f"SUMMARY={summary_path}")
    if previous_sigbreak_handler is not None:
        signal.signal(signal.SIGBREAK, previous_sigbreak_handler)

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
        max_consecutive_dominant_panel_class=int(args.max_consecutive_dominant_panel_class),
        max_consecutive_dominant_escalation_reason=int(args.max_consecutive_dominant_escalation_reason),
        max_consecutive_slow_no_op=int(args.max_consecutive_slow_no_op),
        slow_no_op_threshold_seconds=float(args.slow_no_op_threshold_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
