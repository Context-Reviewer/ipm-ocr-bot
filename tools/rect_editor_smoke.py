from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from tools.rect_editor import OverlayWindow
from src.rect_store import RectStore


def main():
    rects_path = REPO_ROOT / "rects.json"
    store = RectStore.load(rects_path)

    app = QtWidgets.QApplication(sys.argv)
    win = OverlayWindow(store, "BlueStacks App Player")
    win.show()

    # Create a temporary rect and nudge it to ensure no recursion
    win.add_box_named("SMOKE_TEST", (20, 20, 80, 50), select=True)
    box = win.boxes["SMOKE_TEST"]
    box.setRect(20, 20, 100, 60)
    box.request_sync()
    box.setPos(30, 30)
    box.request_sync()

    QtCore.QTimer.singleShot(1000, app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
