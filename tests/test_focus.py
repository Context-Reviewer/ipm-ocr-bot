from ipm.config import FocusConfig
from ipm.focus import title_matches_focus_target


def test_title_matches_focus_target_accepts_expected_window():
    cfg = FocusConfig(window_substring="BlueStacks App Player", excluded_substrings=("Keymap Overlay",))
    assert title_matches_focus_target("BlueStacks App Player", cfg) is True


def test_title_matches_focus_target_rejects_excluded_window():
    cfg = FocusConfig(window_substring="BlueStacks App Player", excluded_substrings=("Keymap Overlay",))
    assert title_matches_focus_target("BlueStacks App Player - Keymap Overlay", cfg) is False
