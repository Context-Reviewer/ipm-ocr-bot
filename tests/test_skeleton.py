import ipm.app as app_module
from ipm.app import Application
from ipm.config import RuntimeConfig, SchedulerConfig
from ipm.scheduler import ScheduledTask, Scheduler


def test_scheduler_marks_due_tasks():
    scheduler = Scheduler(
        [
            ScheduledTask(name="planets", interval_seconds=60.0),
            ScheduledTask(name="ores", interval_seconds=20.0),
        ]
    )
    seeded = scheduler.seed(100.0)
    assert [task.name for task in scheduler.due(100.0, seeded)] == ["planets", "ores"]

    scheduler.mark_complete(ScheduledTask(name="ores", interval_seconds=20.0), 100.0, seeded)
    assert [task.name for task in scheduler.due(100.0, seeded)] == ["planets"]


def test_application_boots_with_stub_tasks():
    cfg = RuntimeConfig(scheduler=SchedulerConfig(tasks={"planets": 60.0, "ores": 20.0}))
    app = Application(cfg)
    assert sorted(app.tasks.keys()) == ["ores", "planets"]
    assert app.capture_backend.name == "desktop"
    assert app.perception_backend.name == "hybrid"


def test_application_tick_runs_stub_tasks(capsys, monkeypatch):
    cfg = RuntimeConfig(scheduler=SchedulerConfig(tasks={"planets": 60.0, "ores": 20.0}))
    app = Application(cfg)
    monkeypatch.setattr(app_module, "ensure_focus", lambda _cfg: True)
    monkeypatch.setattr(app_module, "get_active_window_title", lambda: "BlueStacks App Player")
    app.actions.reset_ui = lambda: None
    app.actions.open_planet_menu = lambda: True
    app.actions.increase_planet_stat = lambda _stat: True
    app.actions.open_ores_panel = lambda: True
    app.actions.select_ore_row = lambda _row: True
    app.actions.open_sell_dialog = lambda: True
    app.actions.choose_sell_fraction = lambda _fraction: True
    app.actions.execute_sell = lambda: True
    app.actions.close_ores_panel = lambda: None
    app.runtime.running = True
    app.tick()
    captured = capsys.readouterr().out
    assert "[TASK] planets start" in captured
    assert "planet_panel" in captured
    assert "verified" in captured
    assert "[TASK] ores start" in captured
    assert "ore_rows" in captured
