from ipm.config import FocusConfig
from ipm.focus import ensure_focus_result, title_matches_focus_target
from window_win32 import ActivationResult


def test_title_matches_focus_target_accepts_expected_window():
    cfg = FocusConfig(window_substring="BlueStacks App Player", excluded_substrings=("Keymap Overlay",))
    assert title_matches_focus_target("BlueStacks App Player", cfg) is True


def test_title_matches_focus_target_rejects_excluded_window():
    cfg = FocusConfig(window_substring="BlueStacks App Player", excluded_substrings=("Keymap Overlay",))
    assert title_matches_focus_target("BlueStacks App Player - Keymap Overlay", cfg) is False


def test_ensure_focus_result_reports_already_focused(monkeypatch):
    cfg = FocusConfig(window_substring="BlueStacks App Player", excluded_substrings=("Keymap Overlay",))
    monkeypatch.setattr("ipm.focus.get_active_window_title", lambda: "BlueStacks App Player")

    result = ensure_focus_result(cfg)

    assert result.ok is True
    assert result.reason == "already_focused"
    assert result.active_title_before == "BlueStacks App Player"
    assert result.active_title_after == "BlueStacks App Player"


def test_ensure_focus_result_reports_window_not_found(monkeypatch):
    cfg = FocusConfig(window_substring="BlueStacks App Player", excluded_substrings=("Keymap Overlay",))
    titles = iter(["PowerShell 7", "PowerShell 7"])
    monkeypatch.setattr("ipm.focus.get_active_window_title", lambda: next(titles))
    monkeypatch.setattr(
        "ipm.focus.activate_window_by_title_substring_result",
        lambda sub: ActivationResult(
            ok=False,
            status="window_not_found",
            hwnd=None,
            active_title_before="PowerShell 7",
            active_title_after="PowerShell 7",
        ),
    )

    result = ensure_focus_result(cfg)

    assert result.ok is False
    assert result.reason == "window_not_found"
    assert result.activation_status == "window_not_found"
    assert result.active_title_before == "PowerShell 7"
    assert result.active_title_after == "PowerShell 7"


def test_ensure_focus_result_reports_foreground_mismatch_after_activation(monkeypatch):
    cfg = FocusConfig(window_substring="BlueStacks App Player", excluded_substrings=("Keymap Overlay",))
    titles = iter(["PowerShell 7", "Google Chrome"])
    monkeypatch.setattr("ipm.focus.get_active_window_title", lambda: next(titles))
    monkeypatch.setattr("ipm.focus.time.sleep", lambda delay: None)
    monkeypatch.setattr(
        "ipm.focus.activate_window_by_title_substring_result",
        lambda sub: ActivationResult(
            ok=False,
            status="foreground_mismatch",
            hwnd=131950,
            active_title_before="PowerShell 7",
            active_title_after="Google Chrome",
            target_title="BlueStacks App Player",
        ),
    )

    result = ensure_focus_result(cfg)

    assert result.ok is False
    assert result.reason == "foreground_mismatch"
    assert result.activation_status == "foreground_mismatch"
    assert result.target_title == "BlueStacks App Player"
    assert result.active_title_before == "PowerShell 7"
    assert result.active_title_after == "Google Chrome"
