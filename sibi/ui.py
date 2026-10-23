"""On-screen rendering. Owner: Sahril.

Interface contract (consumer: app_realtime.py / Gregorius):
    draw(frame, hand, letter, conf, progress, text, fps) -> frame
Draws the hand landmarks, the current letter, a confidence bar, the
hold-to-commit progress indicator, a side text panel, the FPS, and the key
hints. Mutates and returns the frame.
"""
import cv2
import mediapipe as mp

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils

PANEL_W = 220          # right-hand text panel width in px
_HINTS = "Space=spasi  Backspace=hapus  C=clear  Q=keluar"


def _bar(frame, x, y, w, h, frac, color):
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.rectangle(frame, (x, y), (x + int(w * frac), y + h), color, -1)


def _wrap(text, width):
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def draw(frame, hand, letter, conf, progress, text, fps):
    h, w = frame.shape[:2]

    if hand is not None:
        _mp_draw.draw_landmarks(frame, hand, _mp_hands.HAND_CONNECTIONS)

    # Current predicted letter (big).
    cv2.putText(frame, letter or "-", (20, 70), _FONT, 2.2, (0, 255, 0), 3)

    # Confidence bar.
    cv2.putText(frame, "conf", (20, 100), _FONT, 0.5, (200, 200, 200), 1)
    _bar(frame, 70, 88, 160, 14, conf, (0, 180, 255))

    # Hold-to-commit progress.
    cv2.putText(frame, "hold", (20, 125), _FONT, 0.5, (200, 200, 200), 1)
    _bar(frame, 70, 113, 160, 14, progress, (0, 255, 120))

    # FPS.
    cv2.putText(frame, f"{fps:4.1f} FPS", (20, h - 15), _FONT, 0.6, (0, 255, 255), 2)

    # Right-hand text panel.
    x0 = w - PANEL_W
    cv2.rectangle(frame, (x0, 0), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "TEKS", (x0 + 12, 30), _FONT, 0.7, (255, 255, 255), 2)
    y = 70
    for line in _wrap(text, 12):
        cv2.putText(frame, line, (x0 + 12, y), _FONT, 0.8, (255, 255, 255), 2)
        y += 34

    # Key hints (bottom, inside video area).
    cv2.putText(frame, _HINTS, (20, h - 40), _FONT, 0.45, (180, 180, 180), 1)
    return frame
