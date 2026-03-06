from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from PySide6 import QtCore, QtGui, QtWidgets

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from src.window_win32 import get_bluestacks_client_rect, ClientRect
from src.rect_store import RectStore, Rect
from src import ocr, config
from tools.rect_editor_utils import color_for_name, snap_value, snap_rect


DEFAULT_RECTS: Dict[str, Rect] = {
    "ORE_ROW1_QTY": (0, 0, 80, 30),
    "ORE_ROW2_QTY": (0, 40, 80, 30),
    "HUD_CASH": (0, 0, 160, 40),
    "PLANET_STATS_PANEL": (0, 0, 300, 240),
}

HANDLE = 7
MIN_SIZE = 10
RECT_EDITOR_DEBUG = False
AUTO_PREFIX = "RECT_NEW_"


def _text_color_for_bg(rgb: tuple[int, int, int]) -> QtGui.QColor:
    r, g, b = rgb
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return QtGui.QColor(0, 0, 0) if luminance > 0.6 else QtGui.QColor(255, 255, 255)


class ResizeHandle(QtWidgets.QGraphicsRectItem):
    def __init__(self, parent_box: "RectBox", corner: str):
        self.parent_box = parent_box
        self.corner = corner
        super().__init__()
        self.setParentItem(parent_box)
        self.setRect(0, 0, HANDLE, HANDLE)
        self.setZValue(10)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)

    def set_color(self, color: QtGui.QColor):
        pen = QtGui.QPen(color, 1)
        brush = QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 220))
        self.setPen(pen)
        self.setBrush(brush)

    def itemChange(self, change, value):
        parent = getattr(self, "parent_box", None)
        if parent is None:
            return super().itemChange(change, value)
        if getattr(parent, "_syncing", False):
            return super().itemChange(change, value)
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            if RECT_EDITOR_DEBUG:
                print(f"[RECT] handle_move corner={self.corner} pos=({value.x():.1f},{value.y():.1f})")
            parent.handle_moved(self.corner, value)
            return super().itemChange(change, value)
        return super().itemChange(change, value)


class RectBox(QtWidgets.QGraphicsRectItem):
    def __init__(self, name: str, rect: QtCore.QRectF, on_changed=None, on_selected=None, get_snap=None):
        super().__init__(rect)
        self.name = name
        self._syncing = False
        self._sync_pending = False
        self._on_changed = on_changed
        self._on_selected = on_selected
        self._get_snap = get_snap
        self._selected = False

        self.label_bg = QtWidgets.QGraphicsRectItem(self)
        self.label = QtWidgets.QGraphicsSimpleTextItem(name, self)
        self.label.setZValue(11)
        self.label_bg.setZValue(10)

        self.handles = {
            "tl": ResizeHandle(self, "tl"),
            "tr": ResizeHandle(self, "tr"),
            "bl": ResizeHandle(self, "bl"),
            "br": ResizeHandle(self, "br"),
        }

        self.apply_color()
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self._sync_handles()

    def apply_color(self):
        rgb = color_for_name(self.name)
        color = QtGui.QColor(*rgb)
        pen = QtGui.QPen(color, 3 if self._selected else 2)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 35)))
        for h in self.handles.values():
            h.set_color(color)
        text_color = _text_color_for_bg(rgb)
        self.label.setBrush(QtGui.QBrush(text_color))
        bg = QtGui.QColor(color.red(), color.green(), color.blue(), 180)
        self.label_bg.setBrush(QtGui.QBrush(bg))
        self.label_bg.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self._update_label()

    def _update_label(self):
        self.label.setText(self.name)
        self.label.setPos(6, 4)
        bounds = self.label.boundingRect()
        pad_x, pad_y = 4, 2
        self.label_bg.setRect(
            bounds.x() - pad_x,
            bounds.y() - pad_y,
            bounds.width() + pad_x * 2,
            bounds.height() + pad_y * 2,
        )
        self.label_bg.setPos(self.label.pos())

    def set_name(self, name: str):
        self.name = name
        self.apply_color()

    def set_selected(self, selected: bool):
        self._selected = selected
        self.apply_color()

    def _snap_enabled(self) -> tuple[bool, int]:
        if not self._get_snap:
            return (False, 0)
        enabled, grid = self._get_snap()
        if not enabled:
            return (False, grid)
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return (False, grid)
        return (True, grid)

    def _sync_handles(self):
        if self._syncing:
            return
        self._syncing = True
        try:
            r = self.rect()
            self.handles["tl"].setPos(r.left() - HANDLE / 2, r.top() - HANDLE / 2)
            self.handles["tr"].setPos(r.right() - HANDLE / 2, r.top() - HANDLE / 2)
            self.handles["bl"].setPos(r.left() - HANDLE / 2, r.bottom() - HANDLE / 2)
            self.handles["br"].setPos(r.right() - HANDLE / 2, r.bottom() - HANDLE / 2)
        finally:
            self._syncing = False
        self._sync_pending = False

    def request_sync(self):
        if self._syncing or self._sync_pending:
            return
        self._sync_pending = True
        QtCore.QTimer.singleShot(0, self._sync_handles)

    def itemChange(self, change, value):
        if self._syncing:
            return super().itemChange(change, value)
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            enabled, grid = self._snap_enabled()
            if enabled:
                pos = value
                return QtCore.QPointF(snap_value(pos.x(), grid), snap_value(pos.y(), grid))
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.request_sync()
            if self._on_changed:
                self._on_changed()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if self._on_selected:
            self._on_selected(self.name)
        super().mousePressEvent(event)

    def handle_moved(self, corner: str, new_pos: QtCore.QPointF):
        r = self.rect()
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

        enabled, grid = self._snap_enabled()
        if enabled:
            left = snap_value(left, grid)
            top = snap_value(top, grid)
            right = snap_value(right, grid)
            bottom = snap_value(bottom, grid)

        nleft = min(left, right - MIN_SIZE)
        nright = max(right, left + MIN_SIZE)
        ntop = min(top, bottom - MIN_SIZE)
        nbottom = max(bottom, top + MIN_SIZE)

        self.setRect(QtCore.QRectF(nleft, ntop, nright - nleft, nbottom - ntop))
        self.request_sync()
        if self._on_changed:
            self._on_changed()


class RectScene(QtWidgets.QGraphicsScene):
    def __init__(self, controller: "OverlayWindow"):
        super().__init__()
        self.controller = controller
        self._creating = False
        self._create_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self._start_pos: Optional[QtCore.QPointF] = None

    def _item_at(self, pos: QtCore.QPointF):
        item = self.itemAt(pos, QtGui.QTransform())
        while item and not isinstance(item, RectBox):
            item = item.parentItem()
        return item

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.controller.overlay_enabled
            and self._item_at(event.scenePos()) is None
        ):
            self._creating = True
            self._start_pos = event.scenePos()
            if self._create_item:
                self.removeItem(self._create_item)
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 1, QtCore.Qt.PenStyle.DashLine)
            self._create_item = QtWidgets.QGraphicsRectItem()
            self._create_item.setPen(pen)
            self.addItem(self._create_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self._creating and self._create_item and self._start_pos:
            rect = QtCore.QRectF(self._start_pos, event.scenePos()).normalized()
            self._create_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self._creating and self._create_item and self._start_pos:
            rect = self._create_item.rect().normalized()
            self.removeItem(self._create_item)
            self._create_item = None
            self._creating = False
            self._start_pos = None

            if rect.width() < MIN_SIZE or rect.height() < MIN_SIZE:
                event.accept()
                return

            enabled, grid = self.controller.get_snap_settings()
            if enabled:
                x, y, w, h = snap_rect(rect.x(), rect.y(), rect.width(), rect.height(), grid)
            else:
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            name = self.controller.next_auto_name()
            self.controller.add_box_named(name, (int(x), int(y), int(w), int(h)), select=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class OcrPreviewWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, controller: "OverlayWindow"):
        super().__init__(parent)
        self.controller = controller

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["generic", "hud_cash", "ore_qty"])
        self.mode_combo.setCurrentText(getattr(config, "RECT_EDITOR_DEFAULT_MODE", "generic"))
        self.mode_combo.currentTextChanged.connect(self.controller.on_preview_mode_changed)

        self.status_label = QtWidgets.QLabel("Status: FAIL")
        self.raw_label = QtWidgets.QLabel("Raw: ")
        self.parsed_label = QtWidgets.QLabel("Parsed: ")
        self.bbox_label = QtWidgets.QLabel("BBox: ")

        layout.addWidget(QtWidgets.QLabel("OCR Preview"))
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.status_label)
        layout.addWidget(self.raw_label)
        layout.addWidget(self.parsed_label)
        layout.addWidget(self.bbox_label)
        layout.addStretch(1)

    def set_bbox(self, rect: Optional[Rect]):
        if rect is None:
            self.bbox_label.setText("BBox: -")
            return
        x, y, w, h = rect
        self.bbox_label.setText(f"BBox: ({x}, {y}, {w}, {h})")

    def set_result(self, ok: bool, text: str, value: Optional[int], reason: str):
        status = "OK" if ok else f"FAIL ({reason})"
        self.status_label.setText(f"Status: {status}")
        self.raw_label.setText(f"Raw: {text}")
        self.parsed_label.setText(f"Parsed: {value if value is not None else '-'}")


class OverlayWindow(QtWidgets.QMainWindow):
    def __init__(self, store: RectStore, title_hint: str):
        super().__init__()
        self.store = store
        self.title_hint = title_hint
        self.client: Optional[ClientRect] = None
        self._shortcuts = []
        self._save_pending = False
        self.overlay_enabled = True
        self.grid_enabled = True
        self.grid_size = int(getattr(config, "RECT_EDITOR_GRID_SIZE", 5))
        self.ocr_enabled = True
        self.ocr_fps = int(getattr(config, "RECT_EDITOR_OCR_FPS", 4))
        self.selected_name: Optional[str] = None

        self.setWindowTitle("IPM Rect Editor")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.scene = RectScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setStyleSheet("background: transparent;")
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)
        self.view.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCentralWidget(self.view)

        self.boxes: Dict[str, RectBox] = {}
        self._build_boxes()

        self._build_toolbar()
        self._build_left_panel()
        self._build_right_panel()

        self._bind_shortcut("F2", self.toggle_overlay)
        self._bind_shortcut("Ctrl+S", self._save_hotkey)
        self._bind_shortcut("Ctrl+N", self.add_box_prompt)
        self._bind_shortcut("Ctrl+R", self.rename_box)
        self._bind_shortcut("Esc", self.close)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick_anchor)
        self.timer.start(250)

        self.ocr_timer = QtCore.QTimer(self)
        interval = int(1000 / max(1, self.ocr_fps))
        self.ocr_timer.setInterval(interval)
        self.ocr_timer.timeout.connect(self._tick_ocr)
        self.ocr_timer.start()

        self._tick_anchor()
        QtCore.QTimer.singleShot(0, self._ensure_focus)

        print("Rect Editor:")
        print("  F2     toggle overlay")
        print("  Ctrl+S save rects.json")
        print("  Ctrl+N add rect")
        print("  Ctrl+R rename rect")
        print("  Esc    quit")

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("Tools")
        tb.setMovable(False)
        self.addToolBar(tb)

        save_act = QtGui.QAction("Save", self)
        save_act.triggered.connect(self._save_hotkey)
        tb.addAction(save_act)

        reload_act = QtGui.QAction("Reload", self)
        reload_act.triggered.connect(self.reload_rects)
        tb.addAction(reload_act)

        self.grid_act = QtGui.QAction("Grid", self, checkable=True)
        self.grid_act.setChecked(True)
        self.grid_act.triggered.connect(self.toggle_grid)
        tb.addAction(self.grid_act)

        self.overlay_act = QtGui.QAction("Overlay", self, checkable=True)
        self.overlay_act.setChecked(True)
        self.overlay_act.triggered.connect(self.toggle_overlay)
        tb.addAction(self.overlay_act)

        self.ocr_act = QtGui.QAction("OCR Preview", self, checkable=True)
        self.ocr_act.setChecked(True)
        self.ocr_act.triggered.connect(self.toggle_ocr_preview)
        tb.addAction(self.ocr_act)

        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Grid size: "))
        self.grid_combo = QtWidgets.QComboBox()
        self.grid_combo.addItems(["5", "10", "20"])
        self.grid_combo.setCurrentText(str(self.grid_size))
        self.grid_combo.currentTextChanged.connect(self.set_grid_size)
        tb.addWidget(self.grid_combo)

    def _build_left_panel(self):
        dock = QtWidgets.QDockWidget("Rects", self)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self.refresh_list)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.currentTextChanged.connect(self.on_list_selected)
        layout.addWidget(self.search_box)
        layout.addWidget(self.list_widget)
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.refresh_list()

    def _build_right_panel(self):
        dock = QtWidgets.QDockWidget("Preview", self)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.preview = OcrPreviewWidget(dock, self)
        dock.setWidget(self.preview)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _bind_shortcut(self, seq: str, handler):
        sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
        sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(handler)
        self._shortcuts.append(sc)

    def _ensure_focus(self):
        self.raise_()
        self.activateWindow()
        self.setFocus()
        self.view.setFocus()

    def get_snap_settings(self) -> tuple[bool, int]:
        enabled = self.grid_enabled and not (
            QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        )
        return (enabled, self.grid_size)

    def _build_boxes(self):
        if not self.store.rects:
            self.store.rects = dict(DEFAULT_RECTS)
        self.scene.clear()
        self.boxes.clear()
        for name, (x, y, w, h) in sorted(self.store.rects.items()):
            box = RectBox(
                name,
                QtCore.QRectF(x, y, w, h),
                on_changed=self.request_autosave,
                on_selected=self.select_box,
                get_snap=self.get_snap_settings,
            )
            self.scene.addItem(box)
            self.boxes[name] = box
        self.set_overlay_visibility()

    def refresh_list(self, _=None):
        filt = (self.search_box.text() or "").lower()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for name in sorted(self.boxes.keys()):
            if filt and filt not in name.lower():
                continue
            self.list_widget.addItem(name)
        self.list_widget.blockSignals(False)
        if self.selected_name and self.selected_name in self.boxes:
            items = self.list_widget.findItems(self.selected_name, QtCore.Qt.MatchFlag.MatchExactly)
            if items:
                self.list_widget.setCurrentItem(items[0])

    def select_box(self, name: str):
        if name not in self.boxes:
            return
        if self.selected_name and self.selected_name in self.boxes:
            self.boxes[self.selected_name].set_selected(False)
        self.selected_name = name
        self.boxes[name].set_selected(True)
        items = self.list_widget.findItems(name, QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.list_widget.setCurrentItem(items[0])
        self.preview.set_bbox(self.store.rects.get(name))

    def on_list_selected(self, name: str):
        if name:
            self.select_box(name)

    def _tick_anchor(self):
        c = get_bluestacks_client_rect(self.title_hint)
        if not c:
            return
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

    def toggle_grid(self):
        self.grid_enabled = not self.grid_enabled
        self.grid_act.setChecked(self.grid_enabled)

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        self.overlay_act.setChecked(self.overlay_enabled)
        self.set_overlay_visibility()
        if self.overlay_enabled:
            self._ensure_focus()

    def set_overlay_visibility(self):
        for box in self.boxes.values():
            box.setVisible(self.overlay_enabled)

    def toggle_ocr_preview(self):
        self.ocr_enabled = not self.ocr_enabled
        self.ocr_act.setChecked(self.ocr_enabled)

    def set_grid_size(self, text: str):
        try:
            self.grid_size = int(text)
        except ValueError:
            pass

    def on_preview_mode_changed(self, _mode: str):
        pass

    def _tick_ocr(self):
        if not self.ocr_enabled or not self.selected_name:
            return
        name = self.selected_name
        rect = self.store.rects.get(name)
        if rect is None:
            self.preview.set_result(False, "", None, "no_rect")
            return
        mode = self.preview.mode_combo.currentText()
        result = ocr.ocr_read_debug(name, mode=mode)
        self.preview.set_bbox(rect)
        self.preview.set_result(result["ok"], result["text"], result["value"], result["reason"])

    def _save_hotkey(self):
        self.save(silent=False)

    def request_autosave(self):
        if self._save_pending:
            return
        self._save_pending = True
        QtCore.QTimer.singleShot(250, self._autosave)

    def _autosave(self):
        self._save_pending = False
        self.save(silent=True)

    def next_auto_name(self) -> str:
        nums = []
        for name in self.boxes.keys():
            if name.startswith(AUTO_PREFIX):
                tail = name[len(AUTO_PREFIX):]
                if tail.isdigit():
                    nums.append(int(tail))
        next_num = max(nums) + 1 if nums else 1
        return f"{AUTO_PREFIX}{next_num:03d}"

    def add_box_prompt(self):
        default_name = self.next_auto_name()
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "New Rect",
            "Rect name:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not ok:
            return
        name = str(name).strip()
        if not name:
            return
        if name in self.boxes:
            print(f"[ADD] name already exists: {name}")
            return
        self.add_box_named(name, None, select=True)

    def add_box_named(self, name: str, rect: Optional[Rect], select: bool = True):
        if rect is None:
            w, h = 140, 60
            if self.client:
                x = int(round((self.client.width - w) / 2))
                y = int(round((self.client.height - h) / 2))
            else:
                x, y = 10, 10
            rect = (x, y, w, h)
        self.store.rects[name] = rect
        box = RectBox(
            name,
            QtCore.QRectF(rect[0], rect[1], rect[2], rect[3]),
            on_changed=self.request_autosave,
            on_selected=self.select_box,
            get_snap=self.get_snap_settings,
        )
        self.scene.addItem(box)
        self.boxes[name] = box
        self.set_overlay_visibility()
        self.refresh_list()
        if select:
            self.select_box(name)
        self.request_autosave()

    def rename_box(self):
        if not self.boxes:
            return
        names = sorted(self.boxes.keys())
        current, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Rename Rect",
            "Select rect:",
            names,
            0,
            False,
        )
        if not ok:
            return
        current = str(current)
        if current not in self.boxes:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename Rect",
            "New name:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            current,
        )
        if not ok:
            return
        new_name = str(new_name).strip()
        if not new_name or new_name == current:
            return
        if new_name in self.boxes:
            print(f"[RENAME] name already exists: {new_name}")
            return

        box = self.boxes.pop(current)
        box.set_name(new_name)
        self.boxes[new_name] = box
        rect = self.store.rects.pop(current, None)
        if rect is not None:
            self.store.rects[new_name] = rect
        self.refresh_list()
        self.select_box(new_name)
        self.request_autosave()

    def reload_rects(self):
        self.store = RectStore.load(self.store.path)
        self._build_boxes()
        self.refresh_list()

    def save(self, silent: bool = False):
        out: Dict[str, Rect] = {}
        for name, box in self.boxes.items():
            r = box.rect()
            pos = box.pos()
            x = int(round(pos.x() + r.x()))
            y = int(round(pos.y() + r.y()))
            w = int(round(r.width()))
            h = int(round(r.height()))
            out[name] = (x, y, w, h)
        self.store.rects = out
        self.store.save()
        if not silent:
            print(f"[SAVE] wrote {self.store.path} ({len(out)} rects)")


def main():
    rects_path = REPO_ROOT / "rects.json"
    title_hint = "BlueStacks App Player"
    store = RectStore.load(rects_path)

    app = QtWidgets.QApplication(sys.argv)
    win = OverlayWindow(store, title_hint)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
