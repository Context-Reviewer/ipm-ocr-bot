import config
import json


def test_visible_planet_count_uses_telescope_level(monkeypatch):
    monkeypatch.setitem(config.TECH_STATE, "telescope_level", 3)
    assert config.visible_planet_count() == 10


def test_visible_planet_count_never_below_one(monkeypatch):
    monkeypatch.setitem(config.TECH_STATE, "telescope_level", -5)
    assert config.visible_planet_count() == 1


def test_set_visible_planet_count_sets_override_and_telescope_level(monkeypatch):
    monkeypatch.delitem(config.TECH_STATE, "visible_planet_count_override", raising=False)
    monkeypatch.setitem(config.TECH_STATE, "telescope_level", 3)
    config.set_visible_planet_count(13)
    assert config.visible_planet_count() == 13
    assert config.TECH_STATE["telescope_level"] == 4


def test_set_visible_planet_count_normalizes_invalid_progression_counts(monkeypatch):
    monkeypatch.delitem(config.TECH_STATE, "visible_planet_count_override", raising=False)
    monkeypatch.setitem(config.TECH_STATE, "telescope_level", 1)
    config.set_visible_planet_count(9)
    assert config.visible_planet_count() == 10
    assert config.TECH_STATE["visible_planet_count_override"] == 10
    assert config.TECH_STATE["telescope_level"] == 3


def test_visible_planet_count_normalizes_invalid_override(monkeypatch):
    monkeypatch.setitem(config.TECH_STATE, "visible_planet_count_override", 8)
    assert config.visible_planet_count() == 10


def test_effective_visible_planet_count_uses_calibrated_entry_points(monkeypatch):
    monkeypatch.delitem(config.TECH_STATE, "visible_planet_count_override", raising=False)
    monkeypatch.setitem(config.TECH_STATE, "telescope_level", 0)
    monkeypatch.setattr(
        config,
        "get_planet_entry_points",
        lambda: [
            {"point": (10, 10), "planet_id": 1, "verify": "full"},
            {"point": (20, 20), "planet_id": 4, "verify": "full"},
        ],
    )
    assert config.effective_visible_planet_count() == 4


def test_unlocked_ore_row_map_tracks_visible_planets(monkeypatch):
    monkeypatch.setattr(config, "effective_visible_planet_count", lambda: 4)
    assert config.unlocked_ore_row_map() == {
        1: "Copper",
        2: "Iron",
        3: "Lead",
    }
    assert config.visible_ore_rows() == 3


def test_automation_visible_planet_count_tracks_effective_count(monkeypatch):
    monkeypatch.setattr(config, "effective_visible_planet_count", lambda: 7)
    monkeypatch.setattr(config, "calibrated_visible_planet_count", lambda: 4)
    assert config.automation_visible_planet_count() == 7


def test_planet_entry_points_fallback_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PLANET_ENTRY_POINTS_PATH", str(tmp_path / "missing.json"))
    entries = config.get_planet_entry_points()
    assert entries
    assert entries[0]["planet_id"] == 1


def test_planet_entry_points_can_be_saved_and_loaded(monkeypatch, tmp_path):
    path = tmp_path / "planet_entry_points.json"
    monkeypatch.setattr(config, "PLANET_ENTRY_POINTS_PATH", str(path))
    entries = [
        {"point": (111, 222), "planet_id": 1, "verify": "full"},
        {"point": (333, 444), "planet_id": 2, "verify": "full"},
    ]
    assert config.save_planet_entry_points(entries) is True
    loaded = config.get_planet_entry_points()
    assert loaded[0]["point"] == (111, 222)
    assert loaded[1]["planet_id"] == 2
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["point"] == [111, 222]
