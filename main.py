#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEON FaceID — крутое приложение контроля входа на PyQt6 + OpenCV
- Регистрация лиц (30 сэмплов)
- Вход по лицу (LBPH, opencv-contrib)
- Журнал входов, управление пользователями, настройки порога
- Тёмный неоновый UI, анимация сканирования, демо-режим без камеры

Запуск:
    pip install -r requirements.txt
    python main.py
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QRect, QPoint
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction, QPainterPath, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QLineEdit, QProgressBar,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QSlider, QSpinBox, QCheckBox, QFrame, QSplitter, QMessageBox,
    QHeaderView, QAbstractItemView, QGraphicsOpacityEffect, QDialog
)

# ── Пути ──────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
USERS_JSON = DATA_DIR / "users.json"
LOGS_JSON = DATA_DIR / "logs.json"
SETTINGS_JSON = DATA_DIR / "settings.json"
MODEL_YML = DATA_DIR / "lbph.yml"

DATA_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)

DEFAULT_SETTINGS = {"threshold": 75, "camera": 0, "mirror": True}


def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_cascade(filename: str) -> str:
    """Находит haarcascade XML: локально, в cv2.data или скачивает с GitHub (для OpenCV 5)."""
    # 1) уже скачанный в data/ (рядом со скриптом или exe)
    local = DATA_DIR / filename
    if local.exists() and local.stat().st_size > 1000:
        return str(local)
    # 1b) встроенный в exe (PyInstaller onefile)
    if getattr(sys, "frozen", False):
        bundled = Path(sys._MEIPASS) / "data" / filename
        if bundled.exists() and bundled.stat().st_size > 1000:
            return str(bundled)
    # 2) в пакете cv2
    try:
        cand = Path(cv2.data.haarcascades) / filename
        if cand.exists() and cand.stat().st_size > 1000:
            return str(cand)
    except Exception:
        pass
    # 3) скачать с официального репозитория OpenCV
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
    try:
        import urllib.request
        print(f"Скачиваю {filename} ...")
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, str(local))
        if local.exists() and local.stat().st_size > 1000:
            return str(local)
    except Exception as e:
        print(f"Не удалось скачать {filename}: {e}")
    return str(local)


# ── Движок распознавания ──────────────────────────────────────────
class FaceEngine:
    def __init__(self):
        face_xml = ensure_cascade("haarcascade_frontalface_default.xml")
        eye_xml = ensure_cascade("haarcascade_eye.xml")
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        if self.face_cascade.empty():
            print(f"ВНИМАНИЕ: каскад не загрузился: {face_xml}")
        self.has_contrib = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
        self.recognizer = None
        if self.has_contrib:
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
                if MODEL_YML.exists():
                    self.recognizer.read(str(MODEL_YML))
            except Exception as e:
                print("LBPH init fail:", e)
                self.recognizer = None
                self.has_contrib = False
        # fallback: средние лица
        self.templates: dict[str, np.ndarray] = {}
        self.users: dict = load_json(USERS_JSON, {})
        self._load_templates()

    def _load_templates(self):
        self.templates.clear()
        for uid, info in self.users.items():
            folder = FACES_DIR / uid
            if not folder.exists():
                continue
            imgs = []
            for p in sorted(folder.glob("*.png"))[:40]:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs.append(cv2.resize(img, (200, 200)))
            if imgs:
                self.templates[uid] = np.mean(imgs, axis=0).astype(np.uint8)

    def reload_users(self):
        self.users = load_json(USERS_JSON, {})
        self._load_templates()
        if self.has_contrib and self.recognizer is not None and MODEL_YML.exists():
            try:
                self.recognizer.read(str(MODEL_YML))
            except Exception:
                pass

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.face_cascade.empty():
            return gray, []
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
        return gray, faces

    def crop_face(self, gray, rect):
        x, y, w, h = rect
        pad = int(0.12 * w)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = x + w + pad; y2 = y + h + pad
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (200, 200))

    def train(self) -> tuple[bool, str]:
        self.users = load_json(USERS_JSON, {})
        faces, labels = [], []
        uid_to_int, int_to_uid = {}, {}
        for i, uid in enumerate(sorted(self.users.keys())):
            uid_to_int[uid] = i
            int_to_uid[i] = uid
            folder = FACES_DIR / uid
            for p in sorted(folder.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(i)
        if len(faces) < 5:
            return False, "Нужно минимум 5 фото. Зарегистрируйте пользователя."
        # сохранить маппинг
        save_json(DATA_DIR / "labels.json", int_to_uid)
        if self.has_contrib and self.recognizer is not None:
            try:
                self.recognizer.train(faces, np.array(labels))
                self.recognizer.write(str(MODEL_YML))
            except Exception as e:
                return False, f"Ошибка обучения LBPH: {e}"
        self._load_templates()
        return True, f"Модель обучена: {len(faces)} фото, {len(uid_to_int)} чел."

    def predict(self, face200: np.ndarray):
        """Возвращает (uid | None, confidence, score01). Меньше conf = лучше."""
        labels_map = load_json(DATA_DIR / "labels.json", {})
        if self.has_contrib and self.recognizer is not None and MODEL_YML.exists():
            try:
                label, conf = self.recognizer.predict(face200)
                uid = str(labels_map.get(str(label), list(self.users.keys())[label] if self.users else None))
                score = max(0.0, min(1.0, 1.0 - conf / 150.0))
                return uid, float(conf), float(score)
            except Exception:
                pass
        # fallback — сравнение с шаблоном по MSE
        if not self.templates:
            return None, 999.0, 0.0
        best_uid, best_mse = None, float("inf")
        for uid, tmpl in self.templates.items():
            mse = float(np.mean((face200.astype(np.float32) - tmpl.astype(np.float32)) ** 2))
            if mse < best_mse:
                best_mse, best_uid = mse, uid
        conf = best_mse / 40.0  # эвристика
        score = max(0.0, min(1.0, 1.0 - best_mse / 6000.0))
        return best_uid, conf, score


# ── Поток камеры ──────────────────────────────────────────────────
class CameraWorker(QThread):
    frame = pyqtSignal(object)   # np.ndarray BGR
    error = pyqtSignal(str)

    def __init__(self, index=0, mirror=True, demo=False):
        super().__init__()
        self.index = index
        self.mirror = mirror
        self.demo = demo
        self._run = True
        self.cap = None

    def stop(self):
        self._run = False
        self.wait(1500)

    def _demo_frame(self, t):
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # неоновый градиент
        for y in range(h):
            img[y, :, 0] = int(20 + 30 * (y / h) + 10 * np.sin(t * 2 + y * 0.03))
            img[y, :, 1] = int(10 + 25 * (y / h))
            img[y, :, 2] = int(40 + 40 * (1 - y / h))
        # "лицо" — силуэт
        cx = int(w / 2 + 60 * np.sin(t * 0.8))
        cy = int(h / 2 + 20 * np.cos(t * 0.6))
        cv2.ellipse(img, (cx, cy), (110, 140), 0, 0, 360, (120, 120, 130), -1)
        cv2.ellipse(img, (cx, cy), (110, 140), 0, 0, 360, (255, 240, 0), 3)
        cv2.circle(img, (cx - 40, cy - 20), 14, (10, 10, 15), -1)
        cv2.circle(img, (cx + 40, cy - 20), 14, (10, 10, 15), -1)
        cv2.putText(img, "DEMO MODE", (w // 2 - 110, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 240, 0), 2)
        return img

    def run(self):
        if self.demo:
            t0 = time.time()
            while self._run:
                self.frame.emit(self._demo_frame(time.time() - t0))
                self.msleep(40)
            return
        # Windows: CAP_DSHOW быстрее открывает
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            self.error.emit(f"Камера {self.index} не найдена. Включён демо-режим.")
            # fallback в демо
            t0 = time.time()
            while self._run:
                self.frame.emit(self._demo_frame(time.time() - t0))
                self.msleep(40)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                self.msleep(60)
                continue
            if self.mirror:
                f = cv2.flip(f, 1)
            self.frame.emit(f)
            self.msleep(30)
        self.cap.release()


# ── Виджет видео с неоновой рамкой и скан-линией ──────────────────
class VideoDisplay(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(460, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pix = None
        self.faces = []
        self.scan_y = 0.0       # 0..1
        self.scan_on = False
        self.state = "idle"     # idle | ok | fail | scan
        self.setStyleSheet("background:#05070f; border-radius:18px;")

    def set_frame(self, pix: QPixmap, faces):
        self._pix = pix
        self.faces = faces
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # скруглённая маска, чтобы кадр не выходил за углы
        clip = QPainterPath()
        clip.addRoundedRect(self.rect().toRectF(), 18, 18)
        p.setClipPath(clip)
        # фон
        p.fillRect(self.rect(), QColor(5, 7, 15))
        if self._pix is not None:
            # вписать с сохранением пропорций
            target = self.rect().adjusted(6, 6, -6, -6)
            scaled = self._pix.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
            ox, oy = x, y
            sx = scaled.width() / self._pix.width()
            sy = scaled.height() / self._pix.height()
            # рамка + углы
            color = {"idle": QColor(0, 240, 255), "scan": QColor(0, 240, 255),
                     "ok": QColor(0, 255, 136), "fail": QColor(255, 60, 90)}[self.state]
            pen = QPen(color, 3)
            p.setPen(pen)
            p.drawRoundedRect(target, 18, 18)
            for (fx, fy, fw, fh) in self.faces:
                rx = int(ox + fx * sx); ry = int(oy + fy * sy)
                rw = int(fw * sx); rh = int(fh * sy)
                p.setPen(QPen(color, 3))
                L = 26
                # уголки
                for (cx, cy, dx, dy) in [(rx, ry, 1, 1), (rx + rw, ry, -1, 1),
                                         (rx, ry + rh, 1, -1), (rx + rw, ry + rh, -1, -1)]:
                    p.drawLine(cx, cy, cx + dx * L, cy)
                    p.drawLine(cx, cy, cx, cy + dy * L)
            # скан-линия
            if self.scan_on and scaled.width() > 0:
                sy_pos = int(oy + self.scan_y * scaled.height())
                grad = QColor(color); grad.setAlpha(220)
                p.setPen(QPen(grad, 3))
                p.drawLine(ox + 14, sy_pos, ox + scaled.width() - 14, sy_pos)
                glow = QColor(color); glow.setAlpha(40)
                p.fillRect(ox + 14, sy_pos - 14, scaled.width() - 28, 28, glow)
        else:
            p.setPen(QColor(120, 140, 170))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "📷 Нет видеосигнала")
        # неоновое свечение по контуру
        p.setPen(QPen(QColor(0, 240, 255, 40), 1))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 20, 20)
        p.end()


# ── Главное окно ──────────────────────────────────────────────────
STYLESHEET = """
QMainWindow { background: #070b16; }
QWidget { font-family: "Segoe UI", sans-serif; color: #e8eefc; font-size: 14px; }
QToolTip { background: #0e1628; color: #e8eefc; border: 1px solid rgba(0, 240, 255, 0.35); border-radius: 6px; padding: 6px 8px; }

#Sidebar { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0d1326, stop:1 #080b18); border-right: 1px solid #1b2540; }
#Logo { font-size: 22px; font-weight: 800; color: #00f0ff; }
#LogoSub { font-size: 11px; font-weight: 600; color: #7b8bb0; }

QPushButton[class="NavButton"] { text-align: left; padding: 11px 12px; border-radius: 12px; color: #aab8d8; background: transparent; border: 1px solid transparent; font-size: 14px; font-weight: 600; }
QPushButton[class="NavButton"]:hover { background: #16203a; color: #ffffff; }
QPushButton[class="NavButton"][active="true"] { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 rgba(0, 240, 255, 0.20), stop:1 rgba(124, 58, 237, 0.20)); color: #ffffff; border: 1px solid rgba(0, 240, 255, 0.35); }

#Card { background: #101728; border: 1px solid #1e2a4a; border-radius: 18px; }
#CardTitle { font-size: 15px; font-weight: 700; color: #ffffff; }
#CardHint { font-size: 12px; color: #7b8bb0; }

QLineEdit { background: #0b1122; border: 1px solid #223052; border-radius: 12px; padding: 11px 14px; selection-background-color: #00f0ff; selection-color: #04121a; }
QLineEdit:focus { border: 1px solid #00f0ff; }

QPushButton#Primary { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00d4ff, stop:1 #7c3aed); color: #ffffff; font-weight: 800; border: none; border-radius: 13px; padding: 12px 18px; }
QPushButton#Primary:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22e6ff, stop:1 #8b5cf6); }
QPushButton#Primary:disabled { background: #243052; color: #6b7a99; }
QPushButton#Ghost { background: #16203a; border: 1px solid #26355c; border-radius: 13px; padding: 12px 18px; font-weight: 700; }
QPushButton#Ghost:hover { background: #1d2a4d; border-color: #00f0ff; }
QPushButton#Ghost:disabled { color: #6b7a99; }
QPushButton#Danger { background: #2a1220; border: 1px solid rgba(255, 60, 90, 0.45); color: #ff8ba0; border-radius: 12px; padding: 9px 14px; font-weight: 700; }
QPushButton#Danger:hover { background: #3a1626; border-color: rgba(255, 60, 90, 0.8); }
QPushButton#Success { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00ff88, stop:1 #00c8ff); color: #04121a; font-weight: 800; border: none; border-radius: 13px; padding: 12px 18px; }
QPushButton#Success:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #33ffa0, stop:1 #33d6ff); }
QPushButton#Success:disabled { background: #243052; color: #6b7a99; }

QProgressBar { background: #0b1122; border: 1px solid #223052; border-radius: 10px; text-align: center; color: #00f0ff; height: 20px; }
QProgressBar::chunk { border-radius: 9px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00f0ff, stop:1 #7c3aed); }

QListWidget { background: #0b1122; border: 1px solid #1e2a4a; border-radius: 14px; padding: 8px; }
QListWidget::item { padding: 10px; border-radius: 10px; }
QListWidget::item:selected { background: #1a2542; color: #ffffff; }
QListWidget::item:hover { background: #141d36; }

QTableWidget { background: #0b1122; border: 1px solid #1e2a4a; border-radius: 14px; gridline-color: #1a2542; }
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected { background: #1a2542; color: #ffffff; }
QHeaderView::section { background: #131c33; color: #8fa0c4; border: none; border-bottom: 1px solid #1e2a4a; padding: 10px; font-weight: 700; }
QTableCornerButton::section { background: #131c33; border: none; }

QSlider::groove:horizontal { height: 8px; background: #1a2542; border-radius: 4px; }
QSlider::handle:horizontal { width: 20px; margin: -6px 0; border-radius: 10px; background: #00f0ff; }
QSlider::handle:horizontal:hover { background: #ffffff; }

QSpinBox { background: #0b1122; border: 1px solid #223052; border-radius: 12px; padding: 8px 10px; }
QSpinBox:focus { border: 1px solid #00f0ff; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 6px; border: 1px solid #223052; background: #0b1122; }
QCheckBox::indicator:hover { border-color: #00f0ff; }
QCheckBox::indicator:checked { background: #00f0ff; border-color: #00f0ff; }

QMessageBox { background: #0e1628; }
QMessageBox QLabel { color: #e8eefc; }
QMessageBox QPushButton { background: #16203a; border: 1px solid #26355c; border-radius: 10px; padding: 8px 20px; font-weight: 700; min-width: 84px; }
QMessageBox QPushButton:hover { background: #1d2a4d; border-color: #00f0ff; }
QMessageBox QPushButton:default { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00d4ff, stop:1 #7c3aed); color: #ffffff; border: none; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 4px 3px; border: none; }
QScrollBar::handle:vertical { background: #26355c; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: rgba(0, 240, 255, 0.7); }
QScrollBar::add-line:vertical { background: none; border: none; height: 0px; }
QScrollBar::sub-line:vertical { background: none; border: none; height: 0px; }
QScrollBar::add-page:vertical { background: none; }
QScrollBar::sub-page:vertical { background: none; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 3px 4px; border: none; }
QScrollBar::handle:horizontal { background: #26355c; border-radius: 4px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: rgba(0, 240, 255, 0.7); }
QScrollBar::add-line:horizontal { background: none; border: none; width: 0px; }
QScrollBar::sub-line:horizontal { background: none; border: none; width: 0px; }
QScrollBar::add-page:horizontal { background: none; }
QScrollBar::sub-page:horizontal { background: none; }

QLabel#BigStatus { font-size: 20px; font-weight: 800; }
QLabel#Mono { font-family: Consolas, monospace; }
"""


class Toast(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.hide()
        self._token = 0
        self._eff = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._eff)
        self._anim = QPropertyAnimation(self._eff, b"opacity")
        self._anim.setDuration(400)
        self._anim.finished.connect(self._on_faded)

    def show_msg(self, text, ok=True):
        fg, border = ("#00ff88", "rgba(0, 255, 136, 0.55)") if ok else ("#ff5c7a", "rgba(255, 92, 122, 0.55)")
        self._token += 1
        self.setStyleSheet(
            f"background: rgba(14, 22, 40, 0.94); border: 1px solid {border}; "
            f"border-radius: 12px; padding: 12px 18px; font-weight: 700; color: {fg};"
        )
        self.setText(text)
        pw = self.parentWidget().width() if self.parentWidget() else 600
        self.setMinimumWidth(0)
        self.setMaximumWidth(max(240, pw - 60))
        self.adjustSize()
        self.move((pw - self.width()) // 2, 24)
        self.show()
        self.raise_()
        self._anim.stop()
        self._eff.setOpacity(1.0)
        QTimer.singleShot(2600, lambda tok=self._token: self._fade(tok))

    def _fade(self, tok):
        if tok != self._token:
            return
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_faded(self):
        if self._eff.opacity() < 0.05:
            self.hide()


class UnlockDialog(QDialog):
    def __init__(self, parent, name, conf, score):
        super().__init__(parent)
        self.setWindowTitle("Доступ разрешён")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(400, 320)
        self.setModal(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setObjectName("DialogFrame")
        frame.setStyleSheet(
            "#DialogFrame { background: #0a1226; border: 1px solid rgba(0, 240, 255, 0.45); border-radius: 18px; }"
            "#DialogFrame QLabel { background: transparent; border: none; }"
        )
        outer.addWidget(frame)

        lay = QVBoxLayout(frame)
        lay.setContentsMargins(26, 22, 26, 22)
        lay.setSpacing(10)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ok = QLabel("✅")
        ok.setFont(QFont("Segoe UI", 52))
        ok.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ok)
        t = QLabel("ДОСТУП РАЗРЕШЁН")
        t.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        t.setStyleSheet("color:#00ff88; background: transparent;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        info = QLabel(f"{name}\nconf={conf:.1f} • match={score*100:.0f}% • {datetime.now():%H:%M:%S}")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color:#8fa0c4; background: transparent;")
        lay.addWidget(info)
        btn = QPushButton("Открыть 🚀")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00ff88, stop:1 #00c8ff); "
            "color: #04121a; font-weight: 800; border: none; border-radius: 13px; padding: 12px 18px; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #33ffa0, stop:1 #33d6ff); }"
        )
        btn.clicked.connect(self.accept)
        lay.addSpacing(6)
        lay.addWidget(btn)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEON FaceID — контроль входа")
        self.resize(1240, 780)
        self.setMinimumSize(1120, 680)
        self.settings = {**DEFAULT_SETTINGS, **load_json(SETTINGS_JSON, {})}
        self.engine = FaceEngine()
        self.cam: CameraWorker | None = None
        self.demo_mode = False
        self.last_frame = None
        self.scanning_login = False
        self.capturing = False
        self.capture_name = ""
        self.capture_buf: list[np.ndarray] = []
        self.capture_target = 30
        self.vote: dict[str, int] = {}
        self.vote_total = 0
        self.scan_phase = 0.0
        self.current_uid_preview = None

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self._tick_scan)
        self.scan_timer.start(30)

        self._log_status("Готов. Нажмите «Запустить камеру».")
        self.refresh_users()
        self.refresh_logs()
        self._apply_settings_to_ui()

    # ── UI ─────────────────────────────────────────────
    def _nav_btn(self, text):
        b = QPushButton(text)
        b.setProperty("class", "NavButton")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setCheckable(True)
        return b

    def _card(self, title, hint=""):
        card = QFrame()
        card.setObjectName("Card")
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(10)
        t = QLabel(title); t.setObjectName("CardTitle")
        v.addWidget(t)
        if hint:
            h = QLabel(hint); h.setObjectName("CardHint"); h.setWordWrap(True)
            v.addWidget(h)
        return card, v

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # Sidebar
        side = QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(280)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(14, 22, 14, 18)
        sv.setSpacing(8)
        logo = QLabel("◉ NEON FaceID")
        logo.setObjectName("Logo")
        sv.addWidget(logo)
        sub = QLabel("КОНТРОЛЬ ВХОДА • V2.0")
        sub.setObjectName("LogoSub")
        sv.addWidget(sub)
        sv.addSpacing(14)

        self.btn_p0 = self._nav_btn("🏠 Главная / Скан")
        self.btn_p1 = self._nav_btn("➕ Регистрация")
        self.btn_p2 = self._nav_btn("👥 Пользователи")
        self.btn_p3 = self._nav_btn("📜 Журнал входов")
        self.btn_p4 = self._nav_btn("⚙️ Настройки")
        self.navs = [self.btn_p0, self.btn_p1, self.btn_p2, self.btn_p3, self.btn_p4]
        for i, b in enumerate(self.navs):
            b.clicked.connect(lambda _=False, n=i: self.goto(n))
            sv.addWidget(b)
        sv.addStretch(1)
        self.lbl_engine = QLabel("")
        self.lbl_engine.setObjectName("CardHint")
        self.lbl_engine.setWordWrap(True)
        sv.addWidget(self.lbl_engine)
        ver = QLabel("PyQt6 • OpenCV • LBPH\n© 2026 NEON Labs")
        ver.setObjectName("CardHint")
        sv.addWidget(ver)
        main.addWidget(side)

        # Content
        content = QWidget()
        cv = QVBoxLayout(content)
        cv.setContentsMargins(22, 18, 22, 18)
        cv.setSpacing(14)
        main.addWidget(content, 1)

        header = QHBoxLayout()
        self.lbl_title = QLabel("Главная — сканирование лица")
        self.lbl_title.setStyleSheet("font-size:22px; font-weight:800;")
        header.addWidget(self.lbl_title)
        header.addStretch(1)
        self.dot = QLabel("● OFFLINE")
        self.dot.setStyleSheet("color:#ff5c7a; font-weight:800;")
        header.addWidget(self.dot)
        cv.addLayout(header)

        self.stack = QStackedWidget()
        cv.addWidget(self.stack, 1)

        self.toast = Toast(content)

        self._page_scan(cv)
        self._page_register()
        self._page_users()
        self._page_logs()
        self._page_settings()
        self.goto(0)

    def _page_scan(self, cv):
        page = QWidget()
        h = QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)
        self.video = VideoDisplay()
        left.addWidget(self.video, 1)
        row = QHBoxLayout()
        self.b_cam = QPushButton("▶ Запустить камеру"); self.b_cam.setObjectName("Primary")
        self.b_cam.clicked.connect(self.toggle_cam)
        self.b_cam.setMinimumWidth(200)
        self.b_login = QPushButton("🔓 Сканировать вход"); self.b_login.setObjectName("Success")
        self.b_login.clicked.connect(self.toggle_login_scan)
        self.b_login.setMinimumWidth(200)
        self.b_login.setEnabled(False)
        row.addWidget(self.b_cam)
        row.addWidget(self.b_login)
        left.addLayout(row)
        h.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(12)
        card, v = self._card("Статус доступа", "Наведите лицо в рамку и нажмите «Сканировать вход». Нужно ~1 сек стабильного распознавания.")
        card.setMinimumWidth(290)
        self.lbl_status = QLabel("Ожидание…")
        self.lbl_status.setObjectName("BigStatus")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)
        self.lbl_who = QLabel("—")
        self.lbl_who.setStyleSheet("font-size:16px; font-weight:700; color:#00f0ff;")
        self.lbl_who.setWordWrap(True)
        v.addWidget(self.lbl_who)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        v.addWidget(self.bar)
        self.lbl_conf = QLabel("conf: — • match: —")
        self.lbl_conf.setObjectName("Mono")
        self.lbl_conf.setWordWrap(True)
        v.addWidget(self.lbl_conf)
        right.addWidget(card)

        card2, v2 = self._card("Последний вход", "")
        card2.setMinimumWidth(290)
        self.lbl_last = QLabel("Пока никого не пускали 😴")
        self.lbl_last.setWordWrap(True)
        v2.addWidget(self.lbl_last)
        right.addWidget(card2)
        right.addStretch(1)
        h.addLayout(right, 2)
        self.stack.addWidget(page)

    def _page_register(self):
        page = QWidget()
        h = QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(16)
        card, v = self._card("➕ Новый пользователь", "1) Введите имя  2) Запустите камеру  3) Нажмите «Захватить 30 фото» и медленно поворачивайте голову  4) Модель обучится автоматически.")
        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("Например: Котова Анна")
        v.addWidget(self.inp_name)
        row = QHBoxLayout()
        self.b_cap = QPushButton("📸 Захватить 30 фото"); self.b_cap.setObjectName("Primary")
        self.b_cap.clicked.connect(self.start_capture)
        self.b_retrain = QPushButton("🧠 Переобучить"); self.b_retrain.setObjectName("Ghost")
        self.b_retrain.clicked.connect(self.retrain)
        row.addWidget(self.b_cap); row.addWidget(self.b_retrain)
        v.addLayout(row)
        self.cap_bar = QProgressBar(); self.cap_bar.setRange(0, 30); self.cap_bar.setValue(0)
        v.addWidget(self.cap_bar)
        self.lbl_cap = QLabel("Готов к захвату.")
        self.lbl_cap.setObjectName("CardHint")
        self.lbl_cap.setWordWrap(True)
        v.addWidget(self.lbl_cap)
        v.addStretch(1)
        h.addWidget(card, 2)

        card2, v2 = self._card("💡 Советы для точности", "• Хорошее освещение спереди\n• Лицо занимает ~50% кадра\n• 30 фото: прямо / влево / вправо / вверх / улыбка\n• Очки — отдельно лучше снять\n• Порог можно крутить в Настройках")
        v2.addStretch(1)
        h.addWidget(card2, 1)
        self.stack.addWidget(page)

    def _page_users(self):
        page = QWidget()
        h = QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        card, v = self._card("👥 Пользователи", "")
        row = QHBoxLayout()
        self.inp_search = QLineEdit(); self.inp_search.setPlaceholderText("🔍 Поиск…")
        self.inp_search.textChanged.connect(self.refresh_users)
        self.b_del = QPushButton("🗑 Удалить"); self.b_del.setObjectName("Danger")
        self.b_del.clicked.connect(self.delete_user)
        row.addWidget(self.inp_search, 1); row.addWidget(self.b_del)
        v.addLayout(row)
        self.user_list = QListWidget()
        v.addWidget(self.user_list, 1)
        h.addWidget(card, 1)
        self.stack.addWidget(page)

    def _page_logs(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        card, cv = self._card("📜 Журнал входов", "Каждый успешный и неуспешный вход пишется сюда + в data/logs.json")
        row = QHBoxLayout()
        self.b_clear_logs = QPushButton("🧹 Очистить"); self.b_clear_logs.setObjectName("Ghost")
        self.b_clear_logs.clicked.connect(self.clear_logs)
        self.b_export = QPushButton("💾 Экспорт CSV"); self.b_export.setObjectName("Ghost")
        self.b_export.clicked.connect(self.export_csv)
        row.addStretch(1); row.addWidget(self.b_export); row.addWidget(self.b_clear_logs)
        cv.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Время", "Пользователь", "Результат", "conf / match"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        cv.addWidget(self.table, 1)
        v.addWidget(card, 1)
        self.stack.addWidget(page)

    def _page_settings(self):
        page = QWidget()
        h = QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        card, v = self._card("⚙️ Настройки", "")
        v.addWidget(QLabel("Порог LBPH (меньше = строже):"))
        r = QHBoxLayout()
        self.sl_thr = QSlider(Qt.Orientation.Horizontal); self.sl_thr.setRange(30, 120); self.sl_thr.setValue(self.settings["threshold"])
        self.sl_thr.valueChanged.connect(lambda x: (self.lbl_thr.setText(str(x)), self.save_settings()))
        self.lbl_thr = QLabel(str(self.settings["threshold"]))
        self.lbl_thr.setStyleSheet("font-weight:800; color:#00f0ff; font-size:18px;")
        r.addWidget(self.sl_thr, 1); r.addWidget(self.lbl_thr)
        v.addLayout(r)
        v.addWidget(QLabel("Индекс камеры:"))
        self.sp_cam = QSpinBox(); self.sp_cam.setRange(0, 5); self.sp_cam.setValue(self.settings["camera"])
        self.sp_cam.valueChanged.connect(self.save_settings)
        v.addWidget(self.sp_cam)
        self.cb_mirror = QCheckBox("🪞 Зеркалить видео"); self.cb_mirror.setChecked(self.settings["mirror"])
        self.cb_mirror.stateChanged.connect(self.save_settings)
        v.addWidget(self.cb_mirror)
        row = QHBoxLayout()
        b_reset = QPushButton("💥 Сбросить всё"); b_reset.setObjectName("Danger")
        b_reset.clicked.connect(self.reset_all)
        b_train = QPushButton("🧠 Переобучить модель"); b_train.setObjectName("Ghost")
        b_train.clicked.connect(self.retrain)
        row.addWidget(b_train); row.addWidget(b_reset)
        v.addLayout(row)
        v.addStretch(1)
        h.addWidget(card, 2)
        card2, v2 = self._card("ℹ️ Движок", "")
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        v2.addWidget(self.lbl_info)
        v2.addStretch(1)
        h.addWidget(card2, 1)
        self.stack.addWidget(page)
        self._update_info()

    # ── Навигация / настройки ─────────────────────────
    def goto(self, n):
        self.stack.setCurrentIndex(n)
        titles = ["Главная — сканирование лица", "Регистрация нового лица",
                  "Пользователи", "Журнал входов", "Настройки"]
        self.lbl_title.setText(titles[n])
        for i, b in enumerate(self.navs):
            b.setChecked(i == n)
            b.setProperty("active", "true" if i == n else "false")
            b.style().unpolish(b); b.style().polish(b)

    def save_settings(self):
        self.settings = {"threshold": self.sl_thr.value(), "camera": self.sp_cam.value(),
                         "mirror": self.cb_mirror.isChecked()}
        save_json(SETTINGS_JSON, self.settings)

    def _apply_settings_to_ui(self):
        self.sl_thr.setValue(self.settings.get("threshold", 75))
        self.sp_cam.setValue(self.settings.get("camera", 0))
        self.cb_mirror.setChecked(self.settings.get("mirror", True))

    def _update_info(self):
        n_users = len(load_json(USERS_JSON, {}))
        n_photos = sum(1 for _ in FACES_DIR.rglob("*.png"))
        contrib = "✅ opencv-contrib (LBPH)" if self.engine.has_contrib else "⚠️ fallback — сравнение шаблонов (установите opencv-contrib-python)"
        self.lbl_info.setText(f"Движок: {contrib}\nПользователей: {n_users}\nФото: {n_photos}\nМодель: {'есть ✅' if MODEL_YML.exists() else 'нет — обучите ❌'}\nПапка: {DATA_DIR}")
        self.lbl_engine.setText(f"Движок: {'LBPH ✅' if self.engine.has_contrib else 'Fallback ⚠️'}\nЛюдей: {n_users}")

    # ── Камера ────────────────────────────────────────
    def toggle_cam(self):
        if self.cam is None:
            idx = self.settings.get("camera", 0)
            mir = self.settings.get("mirror", True)
            self.cam = CameraWorker(idx, mir, demo=False)
            self.cam.frame.connect(self.on_frame)
            self.cam.error.connect(self._on_cam_error)
            self.cam.start()
            self.b_cam.setText("⏹ Остановить")
            self.b_login.setEnabled(True)
            self.video.scan_on = True
            self.dot.setText("● ONLINE")
            self.dot.setStyleSheet("color:#00ff88; font-weight:800;")
            self._log_status("Камера запущена…")
        else:
            self.stop_cam()

    def stop_cam(self):
        if self.cam:
            self.cam.stop()
            self.cam = None
        self.b_cam.setText("▶ Запустить камеру")
        self.b_login.setEnabled(False)
        self.video.scan_on = False
        self.scanning_login = False
        self.b_login.setText("🔓 Сканировать вход")
        self.dot.setText("● OFFLINE")
        self.dot.setStyleSheet("color:#ff5c7a; font-weight:800;")

    def _on_cam_error(self, msg):
        self.demo_mode = True
        self.toast.show_msg("📷 " + msg, ok=False)

    def toggle_login_scan(self):
        if self.cam is None:
            self.toast.show_msg("Сначала запустите камеру!", ok=False)
            return
        self.scanning_login = not self.scanning_login
        self.vote.clear(); self.vote_total = 0
        self.b_login.setText("⏳ Сканирую… (стоп)" if self.scanning_login else "🔓 Сканировать вход")
        self.video.state = "scan" if self.scanning_login else "idle"
        self._log_status("Сканирование входа…" if self.scanning_login else "Ожидание…")

    def _tick_scan(self):
        # анимация скан-линии
        if self.video.scan_on:
            self.scan_phase += 0.012
            if self.scan_phase > 1:
                self.scan_phase = 0
            self.video.scan_y = self.scan_phase
            if self.video._pix is not None:
                self.video.update()

    def on_frame(self, frame: np.ndarray):
        self.last_frame = frame
        gray, faces = self.engine.detect(frame)
        # отрисовка
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # масштабируем координаты лиц: detect на полном кадре — ок
        faces_list = [(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in faces]
        self.video.set_frame(pix, faces_list)

        # захват для регистрации
        if self.capturing:
            self._do_capture(gray, faces)
            return
        # распознавание для входа
        if self.scanning_login:
            self._do_login(gray, faces)

    # ── Регистрация ───────────────────────────────────
    def start_capture(self):
        name = self.inp_name.text().strip()
        if len(name) < 2:
            self.toast.show_msg("Введите имя (мин. 2 символа)!", ok=False)
            return
        if self.cam is None:
            self.toast.show_msg("Сначала запустите камеру на Главной!", ok=False)
            self.goto(0)
            return
        self.capture_name = name
        self.capture_buf = []
        self.capturing = True
        self.cap_bar.setValue(0)
        self.lbl_cap.setText(f"Смотрите в камеру: {name}… медленно двигайте головой")
        self.goto(0)
        self.toast.show_msg(f"📸 Захват для «{name}» — смотрите в камеру!", ok=True)

    def _do_capture(self, gray, faces):
        if len(faces) == 0:
            self.lbl_cap.setText("Лицо не видно — подойдите ближе 👀")
            return
        # берём самое большое лицо, не чаще чем раз в ~150мс
        biggest = max(faces, key=lambda r: r[2] * r[3])
        face200 = self.engine.crop_face(gray, biggest)
        if face200 is None:
            return
        # анти-дубликат: пропускаем слишком похожие подряд
        if self.capture_buf:
            diff = float(np.mean(cv2.absdiff(face200, self.capture_buf[-1])))
            if diff < 4 and len(self.capture_buf) % 3 != 0:
                return
        # лёгкое аугментирование не нужно — берём как есть
        self.capture_buf.append(face200)
        n = len(self.capture_buf)
        self.cap_bar.setValue(min(n, self.capture_target))
        self.lbl_cap.setText(f"Захвачено {n}/{self.capture_target}…")
        if n >= self.capture_target:
            self.finish_capture()

    def finish_capture(self):
        self.capturing = False
        users = load_json(USERS_JSON, {})
        uid = f"u{int(time.time())}"
        # уникальность имени
        base = self.capture_name
        names = {v.get("name") for v in users.values()}
        name = base
        k = 2
        while name in names:
            name = f"{base} ({k})"; k += 1
        folder = FACES_DIR / uid
        folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(self.capture_buf):
            cv2.imwrite(str(folder / f"img_{i:03d}.png"), img)
        users[uid] = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
                      "samples": len(self.capture_buf)}
        save_json(USERS_JSON, users)
        self.capture_buf = []
        self.cap_bar.setValue(0)
        self.inp_name.clear()
        ok, msg = self.engine.train()
        self.engine.reload_users()
        self.refresh_users(); self._update_info()
        self.lbl_cap.setText(f"✅ {name} сохранён! {msg}")
        self.toast.show_msg(f"✅ {name} зарегистрирован!", ok=ok)
        self._log_status(f"Зарегистрирован: {name}")

    def retrain(self):
        ok, msg = self.engine.train()
        self.engine.reload_users()
        self._update_info()
        self.toast.show_msg(msg, ok=ok)

    # ── Вход ──────────────────────────────────────────
    def _do_login(self, gray, faces):
        thr = self.settings.get("threshold", 75)
        if len(faces) == 0:
            self._set_status("Покажите лицо…", "—", 0, "conf: —", idle=True)
            self.video.state = "scan"
            return
        biggest = max(faces, key=lambda r: r[2] * r[3])
        face200 = self.engine.crop_face(gray, biggest)
        if face200 is None:
            return
        uid, conf, score = self.engine.predict(face200)
        users = load_json(USERS_JSON, {})
        name = users.get(uid, {}).get("name", "Неизвестный") if uid else "Неизвестный"
        self.lbl_conf.setText(f"conf: {conf:.1f} (порог {thr}) • match: {score*100:.0f}%")
        self.bar.setValue(int(score * 100))
        # голосование по 7 кадрам
        key = uid if (uid and conf < thr) else "__unknown__"
        self.vote[key] = self.vote.get(key, 0) + 1
        self.vote_total += 1
        self.lbl_who.setText(f"👤 {name} …" if key != "__unknown__" else "👤 Распознаю…")
        if self.vote_total >= 7:
            best = max(self.vote, key=self.vote.get)
            n = self.vote[best]
            self.vote.clear(); self.vote_total = 0
            if best != "__unknown__" and n >= 4:
                self._access_granted(best, users[best]["name"], conf, score)
            else:
                self._access_denied(conf, score)

    def _set_status(self, status, who, bar_val, conf_txt, idle=False):
        self.lbl_status.setText(status)
        self.lbl_who.setText(who)
        self.bar.setValue(bar_val)
        self.lbl_conf.setText(conf_txt)

    def _access_granted(self, uid, name, conf, score):
        self.scanning_login = False
        self.b_login.setText("🔓 Сканировать вход")
        self.video.state = "ok"
        self.lbl_status.setText("✅ ДОСТУП РАЗРЕШЁН")
        self.lbl_status.setStyleSheet("color:#00ff88; font-size:20px; font-weight:800;")
        self.lbl_who.setText(f"👤 {name}")
        self.bar.setValue(int(score * 100))
        self.lbl_last.setText(f"✅ {name} • {datetime.now():%d.%m %H:%M:%S} • match {score*100:.0f}%")
        self.add_log(name, "GRANTED", conf, score)
        self.toast.show_msg(f"✅ Добро пожаловать, {name}!", ok=True)
        QTimer.singleShot(2500, lambda: self.lbl_status.setStyleSheet(""))
        dlg = UnlockDialog(self, name, conf, score)
        dlg.exec()
        self.video.state = "idle"

    def _access_denied(self, conf, score):
        self.video.state = "fail"
        self.lbl_status.setText("⛔ ДОСТУП ЗАПРЕЩЁН")
        self.lbl_status.setStyleSheet("color:#ff5c7a; font-size:20px; font-weight:800;")
        self.add_log("Неизвестный", "DENIED", conf, score)
        self.toast.show_msg("⛔ Лицо не распознано", ok=False)
        QTimer.singleShot(1800, lambda: (setattr(self.video, "state", "scan" if self.scanning_login else "idle"),
                                         self.lbl_status.setStyleSheet("")))

    # ── Пользователи / журнал ─────────────────────────
    def refresh_users(self):
        users = load_json(USERS_JSON, {})
        q = self.inp_search.text().strip().lower() if hasattr(self, "inp_search") else ""
        self.user_list.clear()
        for uid, info in sorted(users.items(), key=lambda kv: kv[1].get("name", "")):
            if q and q not in info.get("name", "").lower():
                continue
            n = len(list((FACES_DIR / uid).glob("*.png"))) if (FACES_DIR / uid).exists() else 0
            item = QListWidgetItem(f"👤  {info.get('name')}   •   {n} фото   •   {info.get('created','')}")
            item.setData(Qt.ItemDataRole.UserRole, uid)
            self.user_list.addItem(item)
        self._update_info()

    def delete_user(self):
        it = self.user_list.currentItem()
        if not it:
            self.toast.show_msg("Выберите пользователя!", ok=False)
            return
        uid = it.data(Qt.ItemDataRole.UserRole)
        users = load_json(USERS_JSON, {})
        name = users.get(uid, {}).get("name", uid)
        if QMessageBox.question(self, "Удалить?", f"Удалить {name} и все его фото?") != QMessageBox.StandardButton.Yes:
            return
        users.pop(uid, None)
        save_json(USERS_JSON, users)
        shutil.rmtree(FACES_DIR / uid, ignore_errors=True)
        # переобучить / удалить модель если пусто
        if users:
            self.engine.train()
        else:
            MODEL_YML.unlink(missing_ok=True)
            (DATA_DIR / "labels.json").unlink(missing_ok=True)
        self.engine.reload_users()
        self.refresh_users()
        self.toast.show_msg(f"🗑 {name} удалён", ok=True)

    def add_log(self, name, result, conf, score):
        logs = load_json(LOGS_JSON, [])
        logs.insert(0, {"time": datetime.now().isoformat(timespec="seconds"),
                        "name": name, "result": result,
                        "conf": round(float(conf), 1), "match": round(float(score) * 100, 1)})
        save_json(LOGS_JSON, logs[:500])
        self.refresh_logs()

    def refresh_logs(self):
        logs = load_json(LOGS_JSON, [])
        self.table.setRowCount(len(logs))
        for r, e in enumerate(logs):
            self.table.setItem(r, 0, QTableWidgetItem(e.get("time", "")))
            self.table.setItem(r, 1, QTableWidgetItem(e.get("name", "")))
            res = QTableWidgetItem("✅ GRANTED" if e.get("result") == "GRANTED" else "⛔ DENIED")
            res.setForeground(QColor("#00ff88") if e.get("result") == "GRANTED" else QColor("#ff5c7a"))
            self.table.setItem(r, 2, res)
            self.table.setItem(r, 3, QTableWidgetItem(f"{e.get('conf','')} / {e.get('match','')}%"))

    def clear_logs(self):
        save_json(LOGS_JSON, [])
        self.refresh_logs()

    def export_csv(self):
        import csv
        logs = load_json(LOGS_JSON, [])
        path = BASE_DIR / f"logs_export_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["time", "name", "result", "conf", "match"])
            w.writeheader(); w.writerows(logs)
        self.toast.show_msg(f"💾 Сохранено: {path.name}", ok=True)

    def reset_all(self):
        if QMessageBox.question(self, "Сброс", "Удалить ВСЕХ пользователей, фото, модель и журнал?") != QMessageBox.StandardButton.Yes:
            return
        shutil.rmtree(FACES_DIR, ignore_errors=True)
        FACES_DIR.mkdir(exist_ok=True)
        for p in [USERS_JSON, LOGS_JSON, MODEL_YML, DATA_DIR / "labels.json"]:
            Path(p).unlink(missing_ok=True)
        self.engine.reload_users()
        self.refresh_users(); self.refresh_logs(); self._update_info()
        self.toast.show_msg("💥 Всё сброшено", ok=True)

    def _log_status(self, s):
        self.lbl_status.setText(s)

    def closeEvent(self, e):
        self.stop_cam()
        super().closeEvent(e)


def _apply_dark_palette(app):
    p = QPalette()
    role = QPalette.ColorRole
    grp = QPalette.ColorGroup
    p.setColor(role.Window, QColor("#070b16"))
    p.setColor(role.WindowText, QColor("#e8eefc"))
    p.setColor(role.Base, QColor("#0b1122"))
    p.setColor(role.AlternateBase, QColor("#101728"))
    p.setColor(role.Text, QColor("#e8eefc"))
    p.setColor(role.Button, QColor("#16203a"))
    p.setColor(role.ButtonText, QColor("#e8eefc"))
    p.setColor(role.BrightText, QColor("#ff5c7a"))
    p.setColor(role.Highlight, QColor("#00d4ff"))
    p.setColor(role.HighlightedText, QColor("#04121a"))
    p.setColor(role.ToolTipBase, QColor("#0e1628"))
    p.setColor(role.ToolTipText, QColor("#e8eefc"))
    p.setColor(role.PlaceholderText, QColor("#5b6b8d"))
    for r in (role.WindowText, role.Text, role.ButtonText, role.PlaceholderText):
        p.setColor(grp.Disabled, r, QColor("#6b7a99"))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("NEON FaceID")
    app.setOrganizationName("NEON Labs")
    _apply_dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
