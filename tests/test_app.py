from datetime import datetime

from PIL import Image

from ipm.app import prepare_run_artifact_dir, save_run_frame


def test_prepare_run_artifact_dir_uses_timestamp_format(tmp_path):
    path = prepare_run_artifact_dir(base_dir=str(tmp_path), now=datetime(2026, 3, 9, 21, 30, 15))
    assert path.name == "20260309_213015"
    assert path.exists()
    assert path.is_dir()


def test_save_run_frame_writes_png_to_run_dir(tmp_path):
    image = Image.new("RGB", (16, 12), (10, 20, 30))
    frame_path = save_run_frame(image, output_dir=tmp_path)
    assert frame_path.endswith("frame.png")
    assert (tmp_path / "frame.png").exists()
