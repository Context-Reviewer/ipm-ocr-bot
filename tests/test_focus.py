import window_win32
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


def test_activate_window_result_returns_direct_success_without_attach(monkeypatch):
    attach_attempted = {"value": False}
    monkeypatch.setattr("window_win32.get_foreground_window_title", lambda: "PowerShell 7")
    monkeypatch.setattr("window_win32._get_window_title", lambda hwnd: "BlueStacks App Player")
    monkeypatch.setattr(
        "window_win32._activate_window_direct",
        lambda hwnd, *, active_title_before, target_title: ActivationResult(
            ok=True,
            status="activated",
            hwnd=hwnd,
            active_title_before=active_title_before,
            active_title_after="BlueStacks App Player",
            target_title=target_title,
        ),
    )
    monkeypatch.setattr(
        "window_win32._activate_window_with_attach_thread_input",
        lambda hwnd, *, active_title_before, target_title: attach_attempted.__setitem__("value", True) or ActivationResult(
            ok=False,
            status="activated_via_attach_thread_input",
            hwnd=hwnd,
            active_title_before=active_title_before,
            active_title_after="BlueStacks App Player",
            target_title=target_title,
            used_attach_thread_input=True,
        ),
    )

    result = window_win32.activate_window_result(123)

    assert result.ok is True
    assert result.status == "activated"
    assert result.used_attach_thread_input is False
    assert attach_attempted["value"] is False


def test_run_activation_sequence_reports_direct_setforeground_failure(monkeypatch):
    monkeypatch.setattr(window_win32.win32gui, "IsIconic", lambda hwnd: False)
    monkeypatch.setattr(window_win32.win32gui, "ShowWindow", lambda hwnd, mode: None)
    monkeypatch.setattr(window_win32.win32gui, "BringWindowToTop", lambda hwnd: None)
    monkeypatch.setattr(
        window_win32.win32gui,
        "SetForegroundWindow",
        lambda hwnd: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("window_win32.get_foreground_window_title", lambda: "PowerShell 7")

    result = window_win32._run_activation_sequence(
        123,
        active_title_before="PowerShell 7",
        target_title="BlueStacks App Player",
        success_status="activated",
        mismatch_status="foreground_mismatch",
        setforeground_failed_status="setforeground_failed",
        used_attach_thread_input=False,
    )

    assert result.ok is False
    assert result.status == "setforeground_failed"
    assert result.error_message == "boom"
    assert result.used_attach_thread_input is False


def test_activate_window_result_uses_attach_branch_after_direct_failure(monkeypatch):
    attach_attempted = {"value": False}
    monkeypatch.setattr("window_win32.get_foreground_window_title", lambda: "PowerShell 7")
    monkeypatch.setattr("window_win32._get_window_title", lambda hwnd: "BlueStacks App Player")
    monkeypatch.setattr(
        "window_win32._activate_window_direct",
        lambda hwnd, *, active_title_before, target_title: ActivationResult(
            ok=False,
            status="setforeground_failed",
            hwnd=hwnd,
            active_title_before=active_title_before,
            active_title_after="PowerShell 7",
            target_title=target_title,
            error_message="direct failed",
        ),
    )
    monkeypatch.setattr(
        "window_win32._activate_window_with_attach_thread_input",
        lambda hwnd, *, active_title_before, target_title: attach_attempted.__setitem__("value", True) or ActivationResult(
            ok=True,
            status="activated_via_attach_thread_input",
            hwnd=hwnd,
            active_title_before=active_title_before,
            active_title_after="BlueStacks App Player",
            target_title=target_title,
            used_attach_thread_input=True,
        ),
    )

    result = window_win32.activate_window_result(123)

    assert attach_attempted["value"] is True
    assert result.ok is True
    assert result.status == "activated_via_attach_thread_input"
    assert result.used_attach_thread_input is True


def test_activate_window_with_attach_thread_input_reports_success(monkeypatch):
    attach_calls: list[tuple[int, int, bool]] = []
    monkeypatch.setattr("window_win32._get_current_thread_id", lambda: 10)
    monkeypatch.setattr("window_win32._get_foreground_window_handle", lambda: 20)
    monkeypatch.setattr("window_win32._get_window_thread_id", lambda hwnd: {20: 30, 123: 40}.get(hwnd, 0))
    monkeypatch.setattr("window_win32.get_foreground_window_title", lambda: "PowerShell 7")
    monkeypatch.setattr(
        "window_win32._attach_thread_input_pair",
        lambda src, dst, attach: attach_calls.append((src, dst, attach)) or (True, ""),
    )
    monkeypatch.setattr(
        "window_win32._run_activation_sequence",
        lambda hwnd, **kwargs: ActivationResult(
            ok=True,
            status="activated_via_attach_thread_input",
            hwnd=hwnd,
            active_title_before=kwargs["active_title_before"],
            active_title_after="BlueStacks App Player",
            target_title=kwargs["target_title"],
            used_attach_thread_input=kwargs["used_attach_thread_input"],
        ),
    )

    result = window_win32._activate_window_with_attach_thread_input(
        123,
        active_title_before="PowerShell 7",
        target_title="BlueStacks App Player",
    )

    assert result.ok is True
    assert result.status == "activated_via_attach_thread_input"
    assert result.used_attach_thread_input is True
    assert attach_calls == [
        (10, 30, True),
        (10, 40, True),
        (10, 40, False),
        (10, 30, False),
    ]


def test_activate_window_with_attach_thread_input_reports_attach_failure(monkeypatch):
    attach_calls: list[tuple[int, int, bool]] = []
    monkeypatch.setattr("window_win32._get_current_thread_id", lambda: 10)
    monkeypatch.setattr("window_win32._get_foreground_window_handle", lambda: 20)
    monkeypatch.setattr("window_win32._get_window_thread_id", lambda hwnd: {20: 30, 123: 40}.get(hwnd, 0))
    monkeypatch.setattr("window_win32.get_foreground_window_title", lambda: "PowerShell 7")

    def fake_attach(src, dst, attach):
        attach_calls.append((src, dst, attach))
        if attach_calls == [(10, 30, True)]:
            return False, "attach failed"
        return True, ""

    monkeypatch.setattr("window_win32._attach_thread_input_pair", fake_attach)

    result = window_win32._activate_window_with_attach_thread_input(
        123,
        active_title_before="PowerShell 7",
        target_title="BlueStacks App Player",
    )

    assert result.ok is False
    assert result.status == "attach_thread_input_failed"
    assert result.error_message == "attach failed"
    assert result.used_attach_thread_input is True
    assert attach_calls == [(10, 30, True)]
