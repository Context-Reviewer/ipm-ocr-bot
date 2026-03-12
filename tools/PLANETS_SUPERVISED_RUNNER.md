# Planets Supervised Runner

Use `tools/capture_live_planets_supervised.py` for repeated live `PlanetsTask` sessions with durable artifacts and safe stop conditions.

## Recommended Commands

Conservative capped session:

```powershell
.\.venv\Scripts\python.exe tools\capture_live_planets_supervised.py --max-runs 10 --sleep-seconds 60 --prefix task_supervised_planets
```

Longer unattended session with conservative watchdogs:

```powershell
.\.venv\Scripts\python.exe tools\capture_live_planets_supervised.py --max-runs 0 --sleep-seconds 60 --max-consecutive-focus-unavailable 3 --max-consecutive-task-exceptions 3 --max-consecutive-artifact-write-failures 3 --max-consecutive-dominant-panel-class 6 --max-consecutive-dominant-escalation-reason 6 --max-consecutive-slow-no-op 0 --slow-no-op-threshold-seconds 20 --prefix task_supervised_planets
```

More cautious no-op watchdog trial:

```powershell
.\.venv\Scripts\python.exe tools\capture_live_planets_supervised.py --max-runs 0 --sleep-seconds 60 --max-consecutive-slow-no-op 4 --slow-no-op-threshold-seconds 25 --prefix task_supervised_planets
```

## Stop Reasons

Normal:
- `max_runs_reached`
  A configured capped session finished normally.

Inspection-required:
- `repeated_focus_unavailable`
  Focus could not be established for the configured number of consecutive cycles.
- `repeated_task_exceptions`
  The helper caught task execution exceptions for the configured number of consecutive cycles.
- `repeated_artifact_write_failures`
  Per-cycle artifact writes failed for the configured number of consecutive cycles.
- `repeated_dominant_panel_class`
  The same dominant non-`seed_only` `panel_class` repeated across the configured number of consecutive cycles.
- `repeated_dominant_escalation_reason`
  The same dominant `escalation_reason` repeated across the configured number of consecutive cycles.
- `repeated_slow_no_op_cycles`
  No-op cycles exceeded the configured duration threshold for the configured number of consecutive cycles.

## Session Review

Open the durable summary first:

```powershell
Get-Content out\<session_dir>\session_summary.json
```

Focus on these fields:
- `stop_reason`
  Why the session ended.
- `run_kind_counts`
  Session split between `no_op` and `action_bearing`.
- `action_outcome_counts`
  Session split between `none`, `verified`, and any other action outcomes present.
- `panel_class_counts`
  Aggregate recurring scan classes across the session.
- `escalation_reason_counts`
  Aggregate escalation reasons across the session.
- `slow_panel_class_counts`
  Aggregate slow-panel classes across the session.
- `watchdog_trigger`
  Exact watchdog stop details when a watchdog fired.
- `last_successful_cycle_at`
  Most recent cycle timestamp with no focus failure or task exception.
- `last_action_bearing_cycle_at`
  Most recent action-bearing cycle timestamp, if any.
- `last_exception_at`
  Most recent task exception timestamp, if any.
- `recent_cycles_tail`
  Small bounded tail of recent cycles for quick inspection without opening every `task_run*.json`.

## Review Workflow

1. Check `stop_reason`.
2. If `stop_reason` is `max_runs_reached`, inspect `run_kind_counts` and `action_outcome_counts`.
3. If a watchdog stop fired, inspect `watchdog_trigger` first.
4. Compare `panel_class_counts`, `escalation_reason_counts`, and `slow_panel_class_counts` to see whether one recurring class dominated the session.
5. Open the `artifact_path` from `recent_cycles_tail` or `runs` only when the summary points to something abnormal.

## Conservative Settings

Use these as starting points for unattended sessions:
- `--sleep-seconds 60`
- `--max-consecutive-focus-unavailable 3`
- `--max-consecutive-task-exceptions 3`
- `--max-consecutive-artifact-write-failures 3`
- `--max-consecutive-dominant-panel-class 6`
- `--max-consecutive-dominant-escalation-reason 6`
- `--max-consecutive-slow-no-op 0`

These settings are intentionally not hair-trigger. They stop on repeated abnormal patterns, not a single cycle.
