"""Real-time SIBI fingerspelling -> text. Owner: Gregorius (model: Azka).

    python app_realtime.py

Pipeline per frame: VideoCapture -> flip -> BGR->RGB -> MediaPipe Hands ->
features.normalize_landmarks -> model.predict_proba -> Stabilizer -> ui.draw.

Keys:  Space=spasi  Backspace=hapus huruf  C=kosongkan  Q=keluar
"""
import os
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

from sibi.features import normalize_landmarks
from sibi.stabilizer import Stabilizer
from sibi.ui import draw

MODEL_PATH = os.path.join("models", "sibi_model.joblib")


def load_model(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run train_model.py first.")
    bundle = joblib.load(path)
    return bundle["model"], bundle["labels"]


def predict(model, labels, hand):
    """Return (letter, confidence) from one hand's landmarks."""
    feat = normalize_landmarks(hand).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    i = int(np.argmax(proba))
    return labels[i], float(proba[i])


def main(camera=0):
    model, labels = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {camera}")

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,  # one-hand scope: a 2nd hand is simply ignored
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    stab = Stabilizer(n_frames=8, min_conf=0.7, cooldown=12)

    text = ""
    prev = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand, letter, conf = None, None, 0.0
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]  # edge case: hand out of
                letter, conf = predict(model, labels, hand)  # frame -> None here

            committed = stab.update(letter, conf)
            if committed is not None:
                text += committed

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            frame = draw(frame, hand, letter, conf, stab.progress, text, fps)
            cv2.imshow("SIBI translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key == ord(" "):
                text += " "
            elif key in (8, 127):            # Backspace / Delete
                text = text[:-1]
            elif key in (ord("c"), ord("C")):
                text = ""
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    print(f"Final text: {text!r}")


if __name__ == "__main__":
    main()
