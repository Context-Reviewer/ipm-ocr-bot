import main as main_module


class FakeApp:
    def __init__(self, *, probe_result=0):
        self.calls = []
        self.probe_result = probe_result

    def run_starfield_probe_once(self):
        self.calls.append(("run_starfield_probe_once",))
        return self.probe_result

    def run_forever(self):
        self.calls.append(("run_forever",))


def test_main_runs_starfield_probe_once_when_flag_is_used(monkeypatch):
    app = FakeApp(probe_result=0)
    monkeypatch.setattr(main_module, "build_application", lambda: app)
    result = main_module.main(["--starfield-probe-once"])
    assert result == 0
    assert app.calls == [("run_starfield_probe_once",)]


def test_main_returns_failure_code_when_probe_once_fails(monkeypatch):
    app = FakeApp(probe_result=1)
    monkeypatch.setattr(main_module, "build_application", lambda: app)
    result = main_module.main(["--starfield-probe-once"])
    assert result == 1
    assert app.calls == [("run_starfield_probe_once",)]


def test_main_keeps_normal_runtime_path_without_flag(monkeypatch):
    app = FakeApp()
    monkeypatch.setattr(main_module, "build_application", lambda: app)
    result = main_module.main([])
    assert result == 0
    assert app.calls == [("run_forever",)]
