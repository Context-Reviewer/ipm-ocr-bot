from ipm.actions import ActionDriver
from ipm.rects import RectStore
from ipm.config import RuntimeConfig


def test_action_driver_normalizes_num_keys():
    driver = ActionDriver(RuntimeConfig())
    assert driver._normalize_key("num5") == "numpad 5"
    assert driver._numpad_vk("numpad 5") == 0x65
    assert driver._numpad_vk("ctrl+1") is None
    assert driver._normalize_key("\\") == "\\"


def test_action_driver_close_planet_panel_prefers_rect_click(monkeypatch):
    rects = RectStore(path=None, rects={"PLANET_PANEL_CLOSE": (10, 20, 30, 40)})
    driver = ActionDriver(RuntimeConfig(), rects=rects)
    calls = []
    monkeypatch.setattr(driver, "click_rect_center", lambda key, delay=None: calls.append(("click_rect_center", key, delay)) or True)
    monkeypatch.setattr(driver, "send_key", lambda key, delay=None: calls.append(("send_key", key, delay)) or True)
    assert driver.close_planet_panel() is True
    assert calls == [("click_rect_center", "PLANET_PANEL_CLOSE", driver.config.actions.menu_delay_seconds)]


def test_action_driver_close_planet_panel_falls_back_to_resources_toggle(monkeypatch):
    driver = ActionDriver(RuntimeConfig(), rects=RectStore(path=None, rects={}))
    calls = []
    monkeypatch.setattr(driver, "click_rect_center", lambda key, delay=None: calls.append(("click_rect_center", key, delay)) or False)
    monkeypatch.setattr(driver, "send_key", lambda key, delay=None: calls.append(("send_key", key, delay)) or True)
    assert driver.close_planet_panel() is True
    assert calls == [
        ("click_rect_center", "PLANET_PANEL_CLOSE", driver.config.actions.menu_delay_seconds),
        ("send_key", driver.config.actions.open_resources_key, driver.config.actions.menu_delay_seconds),
        ("send_key", driver.config.actions.open_resources_key, driver.config.actions.menu_delay_seconds),
    ]
