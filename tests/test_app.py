from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from PIL import Image

import ipm.app as app_module
from ipm.app import Application, UIStateCheck, prepare_run_artifact_dir, save_run_frame
from ipm.config import RuntimeConfig
from ipm.focus import FocusResult
from ipm.starfield_probe import PlanetDiscoveryResult, StarfieldCacheSummary, StarfieldProbeResult


class RecordingActions:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def click_rect_center(self, rect_key: str, *, delay: float | None = None) -> bool:
        self.calls.append(("click_rect_center", rect_key, delay))
        return True

    def click_client_point(self, point, *, delay: float | None = None) -> bool:
        self.calls.append(("click_client_point", point, delay))
        return True

    def reset_ui(self) -> None:
        self.calls.append(("reset_ui",))


def _make_app(tmp_path) -> Application:
    app = Application.__new__(Application)
    app.config = RuntimeConfig()
    app.capture_backend = SimpleNamespace()
    app.actions = RecordingActions()
    app.rects = SimpleNamespace(
        get=lambda key: {
            "PLANET_PANEL_CLOSE": (425, 455, 34, 34),
            "PLANET_TITLE": (45, 455, 170, 75),
            "PLANET_PANEL_TEXT": (35, 445, 430, 510),
        }.get(key)
    )
    app.tasks = {
        "planets": SimpleNamespace(
            reader=object(),
            _panel_readable=lambda panel: bool(panel),
            _probe_panel_confirmed=lambda panel: bool(panel),
        )
    }
    app.runtime = None
    app.scheduler = None
    app.perception_backend = None
    app.state_reader = None
    return app


def _starfield_image(size: tuple[int, int] = (600, 1100)) -> Image.Image:
    image = Image.new("RGB", size, (14, 60, 110))
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = (
                min(255, 8 + int(40 * (x / max(1, size[0] - 1)))),
                min(255, 45 + int(90 * (y / max(1, size[1] - 1)))),
                min(255, 90 + int(70 * ((x + y) / max(1, size[0] + size[1] - 2)))),
            )
    return image


def _overlay_image(size: tuple[int, int] = (600, 1100)) -> Image.Image:
    image = _starfield_image(size=size)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 150, 460, 950), fill=(18, 18, 44))
    draw.rectangle((60, 190, 440, 860), fill=(22, 22, 58))
    draw.text((120, 250), "Advanced Furnace", fill=(240, 240, 245))
    draw.ellipse((250, 930, 350, 1030), fill=(72, 72, 82), outline=(200, 200, 205), width=4)
    draw.line((280, 960, 320, 1000), fill=(235, 235, 240), width=6)
    draw.line((320, 960, 280, 1000), fill=(235, 235, 240), width=6)
    return image


def _planet_panel_image(size: tuple[int, int] = (600, 1100)) -> Image.Image:
    image = _starfield_image(size=size)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 340, size[0], size[1] - 40), fill=(22, 18, 52))
    draw.ellipse((48, 458, 108, 518), fill=(255, 176, 62))
    draw.polygon([(122, 488), (138, 468), (138, 508)], fill=(170, 190, 240))
    draw.polygon([(20, 488), (36, 468), (36, 508)], fill=(170, 190, 240))
    draw.text((50, 470), "8. ACHERON", fill=(245, 245, 250))
    draw.text((260, 520), "Resource", fill=(230, 230, 235))
    draw.text((260, 575), "Mining Rate", fill=(235, 235, 240))
    draw.rounded_rectangle((305, 735, 500, 785), radius=14, outline=(0, 220, 255), width=4)
    draw.rounded_rectangle((305, 835, 500, 885), radius=14, outline=(0, 220, 255), width=4)
    draw.rounded_rectangle((305, 935, 500, 985), radius=14, outline=(0, 220, 255), width=4)
    draw.text((395, 748), "$610.65K", fill=(230, 230, 240))
    return image


def test_prepare_run_artifact_dir_uses_timestamp_format(tmp_path):
    path = prepare_run_artifact_dir(base_dir=str(tmp_path), now=datetime(2026, 3, 9, 21, 30, 15))
    assert path.name == "20260309_213015"
    assert path.exists()
    assert path.is_dir()


def test_save_run_frame_writes_png_to_run_dir(tmp_path):
    image = Image.new("RGB", (16, 12), (10, 20, 30))
    frame_path = save_run_frame(image, output_dir=tmp_path)
    assert frame_path.endswith("frame.png")
    assert (tmp_path / "frame.png").exists()


def test_starfield_command_details_helper_preserves_shape():
    summary = StarfieldCacheSummary(
        exact_hit_accepted=1,
        remap_attempted=2,
        remap_skipped=1,
        remap_skipped_reasons=(("orientation_mismatch", 1),),
    )

    details = Application._starfield_command_details(
        summary,
        boundary="starfield_probe_command",
        command_type="starfield_probe",
        phase="post_invocation",
        failure_category="probe",
        failure_reason="panel_not_confirmed",
    )

    assert details == {
        "observability": {
            "command": {
                "boundary": "starfield_probe_command",
                "type": "starfield_probe",
                "phase": "post_invocation",
                "failure_category": "probe",
                "failure_reason": "panel_not_confirmed",
            },
            "probe": {
                "starfield_cache": {
                    "exact_hit_accepted": 1,
                    "exact_hit_rejected": 0,
                    "remap_attempted": 2,
                    "remap_accepted": 0,
                    "remap_skipped": 1,
                    "fallback_to_rediscovery": 0,
                    "cache_refresh_saved": 0,
                    "remap_skipped_reasons": {"orientation_mismatch": 1},
                }
            },
        }
    }


def test_starfield_command_details_emitter_uses_expected_line_shape(capsys):
    Application._emit_starfield_command_details(
        "STARFIELD_PROBE",
        summary=None,
        boundary="starfield_probe_command",
        command_type="starfield_probe",
        phase="preflight",
        failure_category="focus",
        failure_reason="focus_unavailable",
    )

    captured = capsys.readouterr().out.strip()
    assert captured.startswith("[STARFIELD_PROBE] details={")
    assert "'boundary': 'starfield_probe_command'" in captured
    assert "'type': 'starfield_probe'" in captured
    assert "'phase': 'preflight'" in captured
    assert "'failure_category': 'focus'" in captured
    assert "'failure_reason': 'focus_unavailable'" in captured
    assert "'starfield_cache': None" in captured


def test_recover_starfield_succeeds_on_first_close_attempt(tmp_path):
    app = _make_app(tmp_path)
    states = iter(
        [
            UIStateCheck(state="planet_panel_present", detail="panel_controls_visible", panel_visible=True),
            UIStateCheck(state="starfield_ready", detail="ship_detected", starfield_ready=True),
        ]
    )
    original_evaluate = Application._evaluate_ui_state
    original_capture = Application._capture_frame
    Application._evaluate_ui_state = lambda self, image: next(states)  # type: ignore[assignment]
    frames = iter([Image.new("RGB", (600, 1100), "black")])
    Application._capture_frame = lambda self: next(frames)  # type: ignore[assignment]

    try:
        ok, _ = app._recover_starfield(stage="post_open", image=Image.new("RGB", (600, 1100), "black"))
    finally:
        Application._evaluate_ui_state = original_evaluate  # type: ignore[assignment]
        Application._capture_frame = original_capture  # type: ignore[assignment]

    assert ok is True
    assert app.actions.calls == [("click_rect_center", "PLANET_PANEL_CLOSE", app.config.actions.menu_delay_seconds)]


def test_evaluate_ui_state_rejects_non_starfield_overlay(tmp_path):
    app = _make_app(tmp_path)
    state = app._evaluate_ui_state(_overlay_image())
    assert state.state == "overlay_present"
    assert state.detail == "central_modal_overlay"
    assert state.panel_visible is False
    assert state.starfield_ready is False


def test_evaluate_ui_state_keeps_starfield_frame_ready(tmp_path):
    app = _make_app(tmp_path)
    state = app._evaluate_ui_state(_starfield_image())
    assert state.state == "starfield_ready"
    assert state.panel_visible is False
    assert state.starfield_ready is True


def test_evaluate_ui_state_detects_planet_panel_without_close_signal(tmp_path):
    app = _make_app(tmp_path)
    state = app._evaluate_ui_state(_planet_panel_image())
    assert state.state == "planet_panel_present"
    assert state.detail == "panel_controls_visible"
    assert state.panel_visible is True
    assert state.starfield_ready is False


def test_recover_starfield_succeeds_on_expanded_close_fallback(tmp_path):
    app = _make_app(tmp_path)
    states = iter(
        [
            UIStateCheck(state="planet_panel_present", detail="panel_controls_visible", panel_visible=True),
            UIStateCheck(state="planet_panel_present", detail="panel_controls_visible", panel_visible=True),
            UIStateCheck(state="starfield_ready", detail="ship_detected", starfield_ready=True),
        ]
    )
    original_evaluate = Application._evaluate_ui_state
    original_capture = Application._capture_frame
    Application._evaluate_ui_state = lambda self, image: next(states)  # type: ignore[assignment]
    frames = iter(
        [
            Image.new("RGB", (600, 1100), "black"),
            Image.new("RGB", (600, 1100), "black"),
        ]
    )
    Application._capture_frame = lambda self: next(frames)  # type: ignore[assignment]

    try:
        ok, _ = app._recover_starfield(stage="post_open", image=Image.new("RGB", (600, 1100), "black"))
    finally:
        Application._evaluate_ui_state = original_evaluate  # type: ignore[assignment]
        Application._capture_frame = original_capture  # type: ignore[assignment]

    assert ok is True
    assert app.actions.calls[0] == ("click_rect_center", "PLANET_PANEL_CLOSE", app.config.actions.menu_delay_seconds)
    expanded_clicks = [call for call in app.actions.calls if call[0] == "click_client_point"]
    assert len(expanded_clicks) == 4


def test_recover_starfield_overlay_tries_reset_ui_after_safe_dismiss(tmp_path):
    app = _make_app(tmp_path)
    frames = iter([_overlay_image(), _starfield_image()])
    original_capture = Application._capture_frame
    Application._capture_frame = lambda self: next(frames)  # type: ignore[assignment]

    try:
        ok, frame = app._recover_starfield(stage="pre_run", image=_overlay_image())
    finally:
        Application._capture_frame = original_capture  # type: ignore[assignment]

    assert ok is True
    assert frame is not None
    assert app.actions.calls == [
        ("click_client_point", (32, 96), app.config.actions.menu_delay_seconds),
        ("reset_ui",),
    ]


def test_recover_starfield_overlay_tries_panel_close_fallback_after_reset_ui(tmp_path):
    app = _make_app(tmp_path)
    states = iter(
        [
            UIStateCheck(state="overlay_present", detail="central_modal_overlay", panel_visible=False),
            UIStateCheck(state="overlay_present", detail="central_modal_overlay", panel_visible=False),
            UIStateCheck(state="overlay_present", detail="central_modal_overlay", panel_visible=False),
            UIStateCheck(state="starfield_ready", detail="ship_detected", starfield_ready=True),
        ]
    )
    original_evaluate = Application._evaluate_ui_state
    original_capture = Application._capture_frame
    Application._evaluate_ui_state = lambda self, image: next(states)  # type: ignore[assignment]
    frames = iter(
        [
            Image.new("RGB", (600, 1100), "black"),
            Image.new("RGB", (600, 1100), "black"),
            Image.new("RGB", (600, 1100), "black"),
        ]
    )
    Application._capture_frame = lambda self: next(frames)  # type: ignore[assignment]

    try:
        ok, frame = app._recover_starfield(stage="pre_run", image=_overlay_image())
    finally:
        Application._evaluate_ui_state = original_evaluate  # type: ignore[assignment]
        Application._capture_frame = original_capture  # type: ignore[assignment]

    assert ok is True
    assert frame is not None
    assert app.actions.calls == [
        ("click_client_point", (32, 96), app.config.actions.menu_delay_seconds),
        ("reset_ui",),
        ("click_rect_center", "PLANET_PANEL_CLOSE", app.config.actions.menu_delay_seconds),
    ]


def test_recover_starfield_overlay_uses_safe_dismiss_then_reset_ui_and_stays_fail_closed(tmp_path):
    app = _make_app(tmp_path)
    frames = iter([_overlay_image(), _overlay_image(), _overlay_image(), _overlay_image()])
    original_capture = Application._capture_frame
    Application._capture_frame = lambda self: next(frames)  # type: ignore[assignment]

    try:
        ok, frame = app._recover_starfield(stage="pre_run", image=_overlay_image())
    finally:
        Application._capture_frame = original_capture  # type: ignore[assignment]

    assert ok is False
    assert frame is None
    assert app.actions.calls[0:3] == [
        ("click_client_point", (32, 96), app.config.actions.menu_delay_seconds),
        ("reset_ui",),
        ("click_rect_center", "PLANET_PANEL_CLOSE", app.config.actions.menu_delay_seconds),
    ]
    expanded_clicks = [call for call in app.actions.calls if call[0] == "click_client_point"]
    assert len(expanded_clicks) == 5


def test_run_discovery_recovers_before_proceeding(monkeypatch, tmp_path):
    app = _make_app(tmp_path)
    frame = Image.new("RGB", (600, 1100), "black")
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(app_module, "prepare_run_artifact_dir", lambda **kwargs: tmp_path)
    monkeypatch.setattr(Application, "_capture_frame", lambda self: frame)

    recovery_calls: list[str] = []

    def fake_recover(*, stage: str, image):
        recovery_calls.append(stage)
        return True, image

    monkeypatch.setattr(Application, "_recover_starfield", lambda self, *, stage, image: fake_recover(stage=stage, image=image))

    def fake_discover(**kwargs):
        assert kwargs["image"] is frame
        assert kwargs.get("starfield_ready_check") is None
        return PlanetDiscoveryResult(
            ok=True,
            reason="ok",
            target_rank=1,
            target_point=(123, 456),
            planet_title_raw="DHOLEN",
            planet_title_canonical="Dholen",
            returned_to_starfield=True,
        )

    monkeypatch.setattr(app_module, "discover_starfield_planet_by_rank", fake_discover)

    result = app.run_discover_planet_rank_once(1)

    assert result == 0
    assert recovery_calls == ["pre_run"]


def test_run_discovery_fails_closed_when_overlay_not_dismissed(monkeypatch, tmp_path):
    app = _make_app(tmp_path)
    frames = iter([_overlay_image(), _overlay_image(), _overlay_image(), _overlay_image(), _overlay_image()])
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(Application, "_capture_frame", lambda self: next(frames))

    def fail_if_called(**kwargs):
        raise AssertionError("discovery should not run when overlay blocks starfield readiness")

    monkeypatch.setattr(app_module, "discover_starfield_planet_by_rank", fail_if_called)

    result = app.run_discover_planet_rank_once(1)

    assert result == 1


def test_run_discovery_aborts_cleanly_when_recovery_fails(monkeypatch, tmp_path):
    app = _make_app(tmp_path)
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(Application, "_capture_frame", lambda self: Image.new("RGB", (600, 1100), "black"))
    monkeypatch.setattr(Application, "_recover_starfield", lambda self, *, stage, image: (False, None))

    def fail_if_called(**kwargs):
        raise AssertionError("discovery should not run when pre-run recovery fails")

    monkeypatch.setattr(app_module, "discover_starfield_planet_by_rank", fail_if_called)

    result = app.run_discover_planet_rank_once(1)

    assert result == 1


def test_run_starfield_probe_logs_focus_diagnostics_when_focus_unavailable(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=False,
            reason="setforeground_failed",
            active_title_before="PowerShell 7",
            active_title_after="Google Chrome",
            activation_status="setforeground_failed",
            activation_error="error text",
            target_title="BlueStacks App Player",
        ),
    )

    result = app.run_starfield_probe_once()

    captured = capsys.readouterr().out
    assert result == 1
    assert "[STARFIELD_PROBE] result=focus_unavailable" in captured
    assert "reason=setforeground_failed" in captured
    assert "active_title_before='PowerShell 7'" in captured
    assert "active_title_after='Google Chrome'" in captured
    assert "activation_status=setforeground_failed" in captured
    assert "activation_error='error text'" in captured
    assert "target_title='BlueStacks App Player'" in captured
    assert "[STARFIELD_PROBE] details=" in captured
    assert "'boundary': 'starfield_probe_command'" in captured
    assert "'type': 'starfield_probe'" in captured
    assert "'phase': 'preflight'" in captured
    assert "'failure_category': 'focus'" in captured
    assert "'failure_reason': 'focus_unavailable'" in captured
    assert "'starfield_cache': None" in captured


def test_run_starfield_probe_logs_cache_run_rollup(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    frame = Image.new("RGB", (600, 1100), "black")
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(Application, "_capture_frame", lambda self: frame)
    monkeypatch.setattr(Application, "_recover_starfield", lambda self, *, stage, image: (True, image))
    monkeypatch.setattr(
        app_module,
        "try_open_nearest_starfield_candidate",
        lambda **kwargs: StarfieldProbeResult(
            ok=True,
            reason="open_confirmed",
            target_point=(123, 456),
            cache_summary=StarfieldCacheSummary(exact_hit_accepted=1, cache_refresh_saved=1),
        ),
    )

    result = app.run_starfield_probe_once()

    captured = capsys.readouterr().out
    assert result == 0
    assert captured.count("[STARFIELD_CACHE_RUN] boundary=starfield_probe_command") == 1
    assert "exact_hit_accepted=1" in captured
    assert "cache_refresh_saved=1" in captured
    assert "[STARFIELD_PROBE] details=" in captured
    assert "'starfield_cache': {'exact_hit_accepted': 1" in captured
    assert "'cache_refresh_saved': 1" in captured
    assert "'remap_skipped_reasons': {}" in captured


def test_run_starfield_probe_logs_structured_details_on_post_invocation_failure(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    frame = Image.new("RGB", (600, 1100), "black")
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(Application, "_capture_frame", lambda self: frame)
    monkeypatch.setattr(Application, "_recover_starfield", lambda self, *, stage, image: (True, image))
    monkeypatch.setattr(
        app_module,
        "try_open_nearest_starfield_candidate",
        lambda **kwargs: StarfieldProbeResult(
            ok=False,
            reason="panel_not_confirmed",
            cache_summary=StarfieldCacheSummary(exact_hit_rejected=1, fallback_to_rediscovery=1),
        ),
    )

    result = app.run_starfield_probe_once()

    captured = capsys.readouterr().out
    assert result == 1
    assert "[STARFIELD_PROBE] result=panel_not_confirmed" in captured
    assert "[STARFIELD_PROBE] details=" in captured
    assert "'boundary': 'starfield_probe_command'" in captured
    assert "'type': 'starfield_probe'" in captured
    assert "'phase': 'post_invocation'" in captured
    assert "'failure_category': 'probe'" in captured
    assert "'failure_reason': 'panel_not_confirmed'" in captured
    assert "'starfield_cache': {'exact_hit_accepted': 0" in captured
    assert "'exact_hit_rejected': 1" in captured
    assert "'fallback_to_rediscovery': 1" in captured


def test_run_starfield_probe_recovers_before_proceeding(monkeypatch, tmp_path):
    app = _make_app(tmp_path)
    frame = Image.new("RGB", (600, 1100), "black")
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(Application, "_capture_frame", lambda self: frame)

    recovery_calls: list[str] = []

    def fake_recover(*, stage: str, image):
        recovery_calls.append(stage)
        return True, image

    monkeypatch.setattr(Application, "_recover_starfield", lambda self, *, stage, image: fake_recover(stage=stage, image=image))
    monkeypatch.setattr(
        app_module,
        "try_open_nearest_starfield_candidate",
        lambda **kwargs: StarfieldProbeResult(ok=True, reason="open_confirmed", target_point=(123, 456)),
    )

    result = app.run_starfield_probe_once()

    assert result == 0
    assert recovery_calls == ["pre_run"]


def test_run_starfield_probe_logs_structured_details_when_capture_unavailable(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(Application, "_capture_frame", lambda self: None)

    result = app.run_starfield_probe_once()

    captured = capsys.readouterr().out
    assert result == 1
    assert "[STARFIELD_PROBE] result=capture_unavailable" in captured
    assert "[STARFIELD_PROBE] details=" in captured
    assert "'boundary': 'starfield_probe_command'" in captured
    assert "'type': 'starfield_probe'" in captured
    assert "'phase': 'preflight'" in captured
    assert "'failure_category': 'capture'" in captured
    assert "'failure_reason': 'capture_unavailable'" in captured
    assert "'starfield_cache': None" in captured


def test_run_discovery_logs_focus_diagnostics_when_focus_unavailable(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=False,
            reason="foreground_mismatch",
            active_title_before="PowerShell 7",
            active_title_after="Microsoft Edge",
            activation_status="foreground_mismatch",
            activation_error="",
            target_title="BlueStacks App Player",
        ),
    )

    result = app.run_discover_planet_rank_once(1)

    captured = capsys.readouterr().out
    assert result == 1
    assert "[PLANET_DISCOVERY] result=focus_unavailable" in captured
    assert "reason=foreground_mismatch" in captured
    assert "active_title_before='PowerShell 7'" in captured
    assert "active_title_after='Microsoft Edge'" in captured
    assert "activation_status=foreground_mismatch" in captured
    assert "target_title='BlueStacks App Player'" in captured
    assert "[PLANET_DISCOVERY] details=" in captured
    assert "'boundary': 'planet_discovery_command'" in captured
    assert "'type': 'planet_discovery'" in captured
    assert "'phase': 'preflight'" in captured
    assert "'failure_category': 'focus'" in captured
    assert "'failure_reason': 'focus_unavailable'" in captured
    assert "'starfield_cache': None" in captured


def test_run_discovery_logs_structured_details_when_capture_unavailable(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(Application, "_capture_frame", lambda self: None)

    result = app.run_discover_planet_rank_once(1)

    captured = capsys.readouterr().out
    assert result == 1
    assert "[PLANET_DISCOVERY] result=capture_unavailable" in captured
    assert "[PLANET_DISCOVERY] details=" in captured
    assert "'boundary': 'planet_discovery_command'" in captured
    assert "'type': 'planet_discovery'" in captured
    assert "'phase': 'preflight'" in captured
    assert "'failure_category': 'capture'" in captured
    assert "'failure_reason': 'capture_unavailable'" in captured
    assert "'starfield_cache': None" in captured


def test_run_discovery_logs_cache_run_rollup(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    frame = Image.new("RGB", (600, 1100), "black")
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(app_module, "prepare_run_artifact_dir", lambda **kwargs: tmp_path)
    monkeypatch.setattr(Application, "_capture_frame", lambda self: frame)
    monkeypatch.setattr(
        Application,
        "_recover_starfield",
        lambda self, *, stage, image: (True, image) if stage == "pre_run" else (False, None),
    )

    def fake_discover(**kwargs):
        return PlanetDiscoveryResult(
            ok=True,
            reason="ok",
            target_rank=1,
            target_point=(123, 456),
            planet_title_raw="DHOLEN",
            planet_title_canonical="Dholen",
            returned_to_starfield=True,
            cache_summary=StarfieldCacheSummary(
                remap_attempted=1,
                remap_accepted=1,
                cache_refresh_saved=1,
            ),
        )

    monkeypatch.setattr(app_module, "discover_starfield_planet_by_rank", fake_discover)

    result = app.run_discover_planet_rank_once(1)

    captured = capsys.readouterr().out
    assert result == 0
    assert captured.count("[STARFIELD_CACHE_RUN] boundary=planet_discovery_command") == 1
    assert "remap_attempted=1" in captured
    assert "remap_accepted=1" in captured
    assert "cache_refresh_saved=1" in captured
    assert "[PLANET_DISCOVERY] details=" in captured
    assert "'starfield_cache': {'exact_hit_accepted': 0" in captured
    assert "'remap_attempted': 1" in captured
    assert "'remap_accepted': 1" in captured
    assert "'cache_refresh_saved': 1" in captured
    assert "'remap_skipped_reasons': {}" in captured


def test_run_discovery_reports_return_failure_when_post_open_recovery_fails(monkeypatch, tmp_path):
    app = _make_app(tmp_path)
    frame = Image.new("RGB", (600, 1100), "black")
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(app_module, "prepare_run_artifact_dir", lambda **kwargs: tmp_path)
    monkeypatch.setattr(Application, "_capture_frame", lambda self: frame)

    recovery_calls: list[str] = []

    def fake_recover(*, stage: str, image):
        recovery_calls.append(stage)
        if stage == "pre_run":
            return True, image
        return False, None

    monkeypatch.setattr(
        Application,
        "_recover_starfield",
        lambda self, *, stage, image: fake_recover(stage=stage, image=image),
    )

    def fake_discover(**kwargs):
        returned = kwargs["return_to_starfield"]()
        return PlanetDiscoveryResult(
            ok=True,
            reason="ok" if returned else "return_to_starfield_failed",
            target_rank=1,
            target_point=(123, 456),
            planet_title_raw="DHOLEN",
            planet_title_canonical="Dholen",
            returned_to_starfield=returned,
        )

    monkeypatch.setattr(app_module, "discover_starfield_planet_by_rank", fake_discover)

    result = app.run_discover_planet_rank_once(1)

    assert result == 1
    assert recovery_calls == ["pre_run", "post_open"]


def test_run_discovery_logs_structured_details_on_post_invocation_failure(monkeypatch, tmp_path, capsys):
    app = _make_app(tmp_path)
    frame = Image.new("RGB", (600, 1100), "black")
    monkeypatch.setattr(
        app_module,
        "ensure_focus_result",
        lambda focus: FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before="BlueStacks App Player",
            active_title_after="BlueStacks App Player",
        ),
    )
    monkeypatch.setattr(app_module, "prepare_run_artifact_dir", lambda **kwargs: tmp_path)
    monkeypatch.setattr(Application, "_capture_frame", lambda self: frame)
    monkeypatch.setattr(
        Application,
        "_recover_starfield",
        lambda self, *, stage, image: (True, image) if stage == "pre_run" else (False, None),
    )

    def fake_discover(**kwargs):
        returned = kwargs["return_to_starfield"]()
        return PlanetDiscoveryResult(
            ok=False,
            reason="return_to_starfield_failed" if not returned else "ok",
            target_rank=1,
            target_point=(123, 456),
            planet_title_raw="DHOLEN",
            planet_title_canonical="Dholen",
            returned_to_starfield=returned,
            cache_summary=StarfieldCacheSummary(remap_attempted=1, remap_accepted=1, cache_refresh_saved=1),
        )

    monkeypatch.setattr(app_module, "discover_starfield_planet_by_rank", fake_discover)

    result = app.run_discover_planet_rank_once(1)

    captured = capsys.readouterr().out
    assert result == 1
    assert "[PLANET_DISCOVERY] result=return_to_starfield_failed" in captured
    assert "[PLANET_DISCOVERY] details=" in captured
    assert "'boundary': 'planet_discovery_command'" in captured
    assert "'type': 'planet_discovery'" in captured
    assert "'phase': 'post_invocation'" in captured
    assert "'failure_category': 'recovery'" in captured
    assert "'failure_reason': 'return_to_starfield_failed'" in captured
    assert "'starfield_cache': {'exact_hit_accepted': 0" in captured
    assert "'remap_attempted': 1" in captured
    assert "'remap_accepted': 1" in captured
    assert "'cache_refresh_saved': 1" in captured
