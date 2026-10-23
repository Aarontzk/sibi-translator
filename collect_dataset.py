"""Record hand-landmark samples into data/raw/landmarks.csv. Owner: Farel.

Usage:
    python collect_dataset.py --person farel --hand right

Point your hand at the webcam, then press a letter key (A-Y, no J) to save the
current frame's 21 raw landmarks labelled with that letter. The on-screen panel
shows samples per letter. Frames with no hand are skipped automatically.

Keys:  <letter>  record one sample     Q  quit (CSV is flushed on every save)

Protocol (target per letter per person): vary lighting (bright/dim), distance
(near/far) and a little rotation. Coordinates are stored RAW -- normalization
happens only in features.normalize_landmarks (not here).
"""
import argparse
import csv
import os
from collections import Counter

import cv2
import mediapipe as mp

from sibi.features import CSV_COLUMNS, LETTERS, landmarks_to_array

DATA_PATH = os.path.join("data", "raw", "landmarks.csv")


def load_counts(path):
    counts = Counter()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                counts[row["label"]] += 1
    return counts


def counts_text(counts):
    return "  ".join(f"{L}:{counts.get(L, 0)}" for L in LETTERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="recorder name (stored per row)")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    counts = load_counts(DATA_PATH)
    is_new = not os.path.exists(DATA_PATH)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
    )
    draw = mp.solutions.drawing_utils

    with open(DATA_PATH, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if is_new:
            writer.writerow(CSV_COLUMNS)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand = None
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            total = sum(counts.values())
            cv2.putText(frame, f"{args.person}/{args.hand}  total={total}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "press a letter to record, Q to quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, counts_text(counts), (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            cv2.imshow("collect_dataset", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == 255 or hand is None:
                continue
            letter = chr(key).upper()
            if letter not in LETTERS:
                continue
            pts = landmarks_to_array(hand).reshape(-1)  # 63 raw values
            writer.writerow([letter, args.person, args.hand, *pts.tolist()])
            fcsv.flush()
            counts[letter] += 1

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Saved to {DATA_PATH}. Samples per letter:")
    for L in LETTERS:
        print(f"  {L}: {counts.get(L, 0)}")


if __name__ == "__main__":
    main()
