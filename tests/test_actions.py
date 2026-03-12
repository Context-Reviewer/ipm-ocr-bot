from PIL import Image, ImageDraw

from ipm.actions import (
    ActionDriver,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MOVE,
)
from ipm.rects import RectStore
from ipm.config import RuntimeConfig
from window_win32 import ClientRect


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


def _panel_image(rects: RectStore, *, production: bool = False, smelt: bool = False, craft: bool = False):
    image = Image.new("RGB", (595, 1031), "black")
    draw = ImageDraw.Draw(image)
    for key, enabled in (
        ("PRODUCTION", production),
        ("PRODUCTION_SMELT_TAB", smelt),
        ("PRODUCTION_CRAFT_TAB", craft),
        ("RESOURCES", not production),
    ):
        if not enabled:
            continue
        x, y, w, h = rects.get(key)
        draw.rectangle((x + 4, y + 4, x + w - 4, y + h - 4), fill=(12, 12, 12))
        draw.rectangle((x + 10, y + 10, x + w - 10, y + h - 10), fill="white")
    return image


class _StaticCapture:
    def __init__(self, image):
        self.image = image

    def capture_screen(self):
        return self.image.copy()

    def invalidate(self):
        return None


def test_action_driver_click_client_point_uses_sendinput(monkeypatch):
    driver = ActionDriver(RuntimeConfig())
    sent = []
    monkeypatch.setattr("ipm.actions.ensure_focus", lambda cfg: True)
    monkeypatch.setattr("ipm.actions.get_bluestacks_client_rect", lambda title: ClientRect(left=100, top=200, width=595, height=1031))
    monkeypatch.setattr("ipm.actions.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "_send_mouse_input", lambda *, x, y, flags: sent.append((x, y, flags)))

    assert driver.click_client_point((10, 20), delay=0.0) is True
    assert sent == [
        (110, 220, MOUSEEVENTF_MOVE),
        (110, 220, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN),
        (110, 220, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP),
    ]


def test_action_driver_open_smelter_panel_verifies_production_then_switches_tab(monkeypatch):
    rects = RectStore(
        path=None,
        rects={
            "PRODUCTION": (115, 980, 52, 46),
            "RESOURCES": (20, 980, 60, 42),
            "PRODUCTION_SMELT_TAB": (95, 505, 90, 44),
            "PRODUCTION_CRAFT_TAB": (370, 505, 94, 42),
        },
    )
    capture = _StaticCapture(_panel_image(rects, production=False))
    driver = ActionDriver(RuntimeConfig(), rects=rects, capture_backend=capture)
    clicks = []
    sent_keys = []
    monkeypatch.setattr("ipm.actions.time.sleep", lambda *_args, **_kwargs: None)

    def fake_click_rect_center(key, delay=None):
        clicks.append((key, delay))
        if key == "PRODUCTION":
            capture.image = _panel_image(rects, production=True, craft=True)
        elif key == "PRODUCTION_SMELT_TAB":
            capture.image = _panel_image(rects, production=True, smelt=True)
        return True

    monkeypatch.setattr(driver, "click_rect_center", fake_click_rect_center)
    monkeypatch.setattr(driver, "send_key", lambda key, delay=None: sent_keys.append((key, delay)) or True)

    assert driver.open_smelter_panel() is True
    assert clicks == [
        ("PRODUCTION", driver.config.actions.menu_delay_seconds),
        ("PRODUCTION_SMELT_TAB", driver.config.actions.menu_delay_seconds),
    ]
    assert sent_keys == []


def test_action_driver_open_crafter_panel_falls_back_to_key_when_tab_click_unverified(monkeypatch):
    rects = RectStore(
        path=None,
        rects={
            "PRODUCTION": (115, 980, 52, 46),
            "RESOURCES": (20, 980, 60, 42),
            "PRODUCTION_SMELT_TAB": (95, 505, 90, 44),
            "PRODUCTION_CRAFT_TAB": (370, 505, 94, 42),
        },
    )
    capture = _StaticCapture(_panel_image(rects, production=True, smelt=True))
    driver = ActionDriver(RuntimeConfig(), rects=rects, capture_backend=capture)
    clicks = []
    sent_keys = []
    monkeypatch.setattr("ipm.actions.time.sleep", lambda *_args, **_kwargs: None)

    def fake_click_rect_center(key, delay=None):
        clicks.append((key, delay))
        return key != "PRODUCTION_CRAFT_TAB"

    def fake_send_key(key, delay=None):
        sent_keys.append((key, delay))
        if key == driver.config.actions.craft_tab_key:
            capture.image = _panel_image(rects, production=True, craft=True)
        return True

    monkeypatch.setattr(driver, "click_rect_center", fake_click_rect_center)
    monkeypatch.setattr(driver, "send_key", fake_send_key)

    assert driver.open_crafter_panel() is True
    assert clicks == [("PRODUCTION_CRAFT_TAB", driver.config.actions.menu_delay_seconds)]
    assert sent_keys == [(driver.config.actions.craft_tab_key, driver.config.actions.menu_delay_seconds)]
