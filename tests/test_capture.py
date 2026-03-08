import time

from PIL import Image

from ipm.capture import AdbCaptureBackend, DesktopCaptureBackend


def test_desktop_capture_client_bbox_uses_cached_screen():
    backend = DesktopCaptureBackend(cache_ttl_seconds=999.0)
    backend._cached_screen = Image.new("RGB", (400, 300), "white")
    backend._cached_at = time.monotonic()
    cropped = backend.capture_client_bbox((10, 20, 30, 40))
    assert cropped is not None
    assert cropped.size == (30, 40)


def test_adb_capture_client_bbox_scales_to_target_resolution():
    backend = AdbCaptureBackend(target_resolution=(200, 100), cache_ttl_seconds=999.0)
    backend._cached_screen = Image.new("RGB", (400, 200), "white")
    backend._cached_at = time.monotonic()
    cropped = backend.capture_client_bbox((10, 10, 20, 10))
    assert cropped is not None
    assert cropped.size == (40, 20)


def test_capture_backends_invalidate_cached_screen():
    desktop = DesktopCaptureBackend(cache_ttl_seconds=999.0)
    desktop._cached_screen = Image.new("RGB", (100, 100), "white")
    desktop._cached_at = time.monotonic()
    desktop.invalidate()
    assert desktop._cached_screen is None
    assert desktop._cached_at == 0.0

    adb = AdbCaptureBackend(cache_ttl_seconds=999.0)
    adb._cached_screen = Image.new("RGB", (100, 100), "white")
    adb._cached_at = time.monotonic()
    adb.invalidate()
    assert adb._cached_screen is None
    assert adb._cached_at == 0.0
