from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from src.window_win32 import get_bluestacks_client_rect, ClientRect
from src.rect_store import RectStore, Rect

DEFAULT_RECTS: Dict[str, Rect] = {
    "ORE_ROW1_QTY": (0, 0, 80, 30),
    "ORE_ROW2_QTY": (0, 40, 80, 30),
    "HUD_CASH": (0, 0, 160, 40),
    "PLANET_STATS_PANEL": (0, 0, 300, 240),
}


HANDLE = 7


class ResizeHandle(QtWidgets.QGraphicsRectItem):
    def __init__(self, parent_box: "RectBox", corner: str):
        super().__init__(0, 0, HANDLE, HANDLE)
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 220)))
        self.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 220)))
        self.setZValue(10)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.parent_box = parent_box
        self.corner = corner

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            self.parent_box.handle_moved(self.corner, value)
            return self.pos()
        return super().itemChange(change, value)


class RectBox(QtWidgets.QGraphicsRectItem):
    def __init__(self, name: str, rect: QtCore.QRectF):
        super().__init__(rect)
        self.name = name

        pen = QtGui.QPen(QtGui.QColor(0, 255, 255, 220), 2)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 255, 35)))

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)

        self.label = QtWidgets.QGraphicsSimpleTextItem(name, self)
        self.label.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 230)))
        self.label.setPos(4, 4)
        self.label.setZValue(11)

        self.handles = {
            "tl": ResizeHandle(self, "tl"),
            "tr": ResizeHandle(self, "tr"),
            "bl": ResizeHandle(self, "bl"),
            "br": ResizeHandle(self, "br"),
        }
        for h in self.handles.values():
            h.setParentItem(self)

        self._sync_handles()

    def _sync_handles(self):
        r = self.rect()
        self.handles["tl"].setPos(r.left() - HANDLE / 2, r.top() - HANDLE / 2)
        self.handles["tr"].setPos(r.right() - HANDLE / 2, r.top() - HANDLE / 2)
        self.handles["bl"].setPos(r.left() - HANDLE / 2, r.bottom() - HANDLE / 2)
        self.handles["br"].setPos(r.right() - HANDLE / 2, r.bottom() - HANDLE / 2)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._sync_handles()
        return super().itemChange(change, value)

    def handle_moved(self, corner: str, new_pos: QtCore.QPointF):
        r = self.rect()

        # new_pos is handle-local coords (parented), convert to rect space
        x = float(new_pos.x() + HANDLE / 2)
        y = float(new_pos.y() + HANDLE / 2)

        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()

        if corner == "tl":
            left, top = x, y
        elif corner == "tr":
            right, top = x, y
        elif corner == "bl":
            left, bottom = x, y
        elif corner == "br":
            right, bottom = x, y

        # normalize + minimum size
        nleft = min(left, right - 10)
        nright = max(right, left + 10)
        ntop = min(top, bottom - 10)
        nbottom = max(bottom, top + 10)

        self.setRect(QtCore.QRectF(nleft, ntop, nright - nleft, nbottom - ntop))
        self._sync_handles()


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, store: RectStore, title_hint: str):
        super().__init__()
        self.store = store
        self.title_hint = title_hint
        self.client: Optional[ClientRect] = None

        self.setWindowTitle("IPM Rect Editor")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.view = QtWidgets.QGraphicsView()
        self.view.setStyleSheet("background: transparent;")
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)

        self.scene = QtWidgets.QGraphicsScene()
        self.view.setScene(self.scene)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self.boxes: Dict[str, RectBox] = {}
        self._build_boxes()

        # poll anchor position
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick_anchor)
        self.timer.start(250)

        # hotkeys
        QtGui.QShortcut(QtGui.QKeySequence("F2"), self, activated=self.toggle_visible)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self.save)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)

        self._tick_anchor()

        # show instructions in console (deterministic)
        print("Rect Editor:")
        print("  F2     toggle overlay")
        print("  Ctrl+S save rects.json")
        print("  Esc    quit")

    def toggle_visible(self):
        self.setVisible(not self.isVisible())

    def _build_boxes(self):
        if not self.store.rects:
            self.store.rects = dict(DEFAULT_RECTS)

        self.scene.clear()
        self.boxes.clear()

        for name, (x, y, w, h) in sorted(self.store.rects.items()):
            box = RectBox(name, QtCore.QRectF(x, y, w, h))
            self.scene.addItem(box)
            self.boxes[name] = box

    def _tick_anchor(self):
        c = get_bluestacks_client_rect(self.title_hint)
        if not c:
            return

        # Only move/resize overlay if changed
        if (
            self.client is None
            or c.left != self.client.left
            or c.top != self.client.top
            or c.width != self.client.width
            or c.height != self.client.height
        ):
            self.client = c
            self.setGeometry(c.left, c.top, c.width, c.height)
            self.scene.setSceneRect(0, 0, c.width, c.height)

    def save(self):
        # Persist rects as ints (relative to client origin)
        out: Dict[str, Rect] = {}
        for name, box in self.boxes.items():
            r = box.rect()
            # rect() is local; add item position for moved boxes
            pos = box.pos()
            x = int(round(pos.x() + r.x()))
            y = int(round(pos.y() + r.y()))
            w = int(round(r.width()))
            h = int(round(r.height()))
            out[name] = (x, y, w, h)

        self.store.rects = out
        self.store.save()
        print(f"[SAVE] wrote {self.store.path} ({len(out)} rects)")


def main():
    rects_path = "rects.json"
    title_hint = "BlueStacks App Player"

    store = RectStore.load(rects_path)

    app = QtWidgets.QApplication(sys.argv)
    win = OverlayWindow(store, title_hint)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
