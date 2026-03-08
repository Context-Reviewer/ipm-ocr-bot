from pathlib import Path

import runtime_state


def test_clear_runtime_state_removes_file(monkeypatch, tmp_path):
    state_path = tmp_path / "runtime_state.json"
    monkeypatch.setattr(runtime_state.config, "RUNTIME_STATE_PATH", str(state_path))
    state_path.write_text("{}", encoding="utf-8")

    assert runtime_state.clear_runtime_state() is True
    assert not state_path.exists()
