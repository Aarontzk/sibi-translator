"""FPS and per-stage timing benchmark. Owner: Rafi.

    python benchmark_fps.py --seconds 60

Runs the live pipeline for N seconds and reports average FPS plus mean time per
stage (MediaPipe hand detection vs. model inference). Run on >= 2 laptops and
record both for Bab 7.
"""
import argparse
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

from app_realtime import MODEL_PATH, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    model, labels = bundle["model"], bundle["labels"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
    )

    frames = 0
    t_mp = t_model = 0.0
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        result = hands.process(rgb)
        t1 = time.perf_counter()
        if result.multi_hand_landmarks:
            predict(model, labels, result.multi_hand_landmarks[0])
        t2 = time.perf_counter()

        t_mp += t1 - t0
        t_model += t2 - t1
        frames += 1

    cap.release()
    hands.close()

    elapsed = time.perf_counter() - start
    if frames == 0:
        raise SystemExit("No frames captured.")
    print(f"Frames:           {frames}")
    print(f"Duration:         {elapsed:.1f} s")
    print(f"Average FPS:      {frames / elapsed:.1f}")
    print(f"MediaPipe:        {t_mp / frames * 1000:.2f} ms/frame")
    print(f"Model inference:  {t_model / frames * 1000:.2f} ms/frame")


if __name__ == "__main__":
    main()
