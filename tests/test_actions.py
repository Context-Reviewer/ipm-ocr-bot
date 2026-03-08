from ipm.actions import ActionDriver
from ipm.config import RuntimeConfig


def test_action_driver_normalizes_num_keys():
    driver = ActionDriver(RuntimeConfig())
    assert driver._normalize_key("num5") == "numpad 5"
    assert driver._numpad_vk("numpad 5") == 0x65
    assert driver._numpad_vk("ctrl+1") is None
    assert driver._normalize_key("\\") == "\\"
