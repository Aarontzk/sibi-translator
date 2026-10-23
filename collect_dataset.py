"""Record SIBI hand-landmark samples into data/raw/landmarks.csv. Owner: Farel.

Usage:
    python collect_dataset.py --person farel --hand right [--target 100]

Guided flow: the window shows one letter at a time next to its picture from the
SIBI guide (sibi/guide.py, the same poster as the web page). Copy the handshape,
press Space to start recording, and keep the shape while slowly moving the hand:
nearer/farther, a little tilt, left/right of the frame. Recording stops by itself
at --target samples and jumps to the next letter you have not finished.
Re-running with the same --person continues where you left off.

Keys:  Space  start/pause recording     N  next letter     B  previous letter
       Q      quit (CSV is flushed on every save)

Coordinates are stored RAW -- normalization happens only in
features.normalize_landmarks (not here).
"""
import argparse
import csv
import os
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

from sibi.features import CSV_COLUMNS, LETTERS, landmarks_to_array
from sibi.guide import load_guide

DATA_PATH = os.path.join("data", "raw", "landmarks.csv")
RECORD_EVERY = 2   # save every 2nd frame: neighbouring frames are near-duplicates
PANEL_W = 260
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE, _GREY, _GREEN, _RED = (255, 255, 255), (170, 170, 170), (80, 200, 80), (60, 60, 230)


def load_counts(path, person):
    """Samples per letter already recorded by this person."""
    counts = Counter()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["person"] == person:
                    counts[row["label"]] += 1
    return counts


def next_unfinished(counts, target, start=0):
    """Index of the first letter (from start, wrapping) below target, or None."""
    for k in range(len(LETTERS)):
        i = (start + k) % len(LETTERS)
        if counts[LETTERS[i]] < target:
            return i
    return None


def draw_panel(frame, letter, guide_img, count, target, counts, recording, has_hand):
    """Camera frame + right-hand panel with the letter, its SIBI picture and progress."""
    h = frame.shape[0]
    panel = np.full((h, PANEL_W, 3), 30, np.uint8)
    cv2.putText(panel, letter, (20, 95), _FONT, 2.6, _WHITE, 5)
    gh = 0
    if guide_img is not None:
        gw = PANEL_W - 110
        gh = min(int(guide_img.shape[0] * gw / guide_img.shape[1]), h // 3)
        panel[12:12 + gh, 100:100 + gw] = cv2.resize(guide_img, (gw, gh))
    y = max(130, 12 + gh + 30)
    cv2.putText(panel, f"{count}/{target} sampel", (20, y), _FONT, 0.7, _WHITE, 2)
    cv2.rectangle(panel, (20, y + 12), (PANEL_W - 20, y + 26), (80, 80, 80), -1)
    cv2.rectangle(panel, (20, y + 12), (20 + int((PANEL_W - 40) * min(count / target, 1)), y + 26), _GREEN, -1)
    status, color = ("MEREKAM", _RED) if recording else ("JEDA - tekan Space", _GREY)
    if recording and not has_hand:
        status, color = "tangan tidak terlihat", _GREY
    cv2.putText(panel, status, (20, y + 60), _FONT, 0.6, color, 2)

    done = sum(counts[L] >= target for L in LETTERS)
    cv2.putText(panel, f"selesai {done}/{len(LETTERS)} huruf", (20, y + 100), _FONT, 0.55, _WHITE, 1)
    for i, L in enumerate(LETTERS):  # done letters green, current white
        c = _WHITE if L == letter else (_GREEN if counts[L] >= target else (110, 110, 110))
        cv2.putText(panel, L, (20 + (i % 8) * 28, y + 135 + (i // 8) * 26), _FONT, 0.6, c, 2)
    for j, hint in enumerate(["Space rekam/jeda", "N lanjut  B kembali", "Q keluar"]):
        cv2.putText(panel, hint, (20, h - 70 + j * 22), _FONT, 0.5, _GREY, 1)
    return np.hstack([frame, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="recorder name (stored per row)")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--target", type=int, default=100, help="samples per letter for this person")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    if args.target <= 0:
        raise SystemExit("--target must be positive")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    counts = load_counts(DATA_PATH, args.person)
    is_new = not os.path.exists(DATA_PATH)
    guide = load_guide()
    idx = next_unfinished(counts, args.target)
    if idx is None:
        print(f"{args.person} already has {args.target}+ samples for every letter. "
              f"Use a higher --target to record more.")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
    )
    draw = mp.solutions.drawing_utils
    recording, frame_no = False, 0

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
            hand = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None
            letter = LETTERS[idx]

            frame_no += 1
            if recording and hand is not None and frame_no % RECORD_EVERY == 0:
                pts = landmarks_to_array(hand).reshape(-1)  # 63 raw values
                writer.writerow([letter, args.person, args.hand, *pts.tolist()])
                fcsv.flush()
                counts[letter] += 1
                if counts[letter] >= args.target:
                    recording = False
                    nxt = next_unfinished(counts, args.target, idx + 1)
                    if nxt is None:
                        print("All letters done.")
                        break
                    idx = nxt

            if hand is not None:
                draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            view = draw_panel(frame, letter, guide.get(letter), counts[letter], args.target,
                              counts, recording, hand is not None)
            cv2.imshow("collect_dataset", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                recording = not recording
            elif key in (ord("n"), ord("N")):
                recording, idx = False, (idx + 1) % len(LETTERS)
            elif key in (ord("b"), ord("B")):
                recording, idx = False, (idx - 1) % len(LETTERS)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Saved to {DATA_PATH}. Samples per letter for {args.person}:")
    print("  " + "  ".join(f"{L}:{counts[L]}" for L in LETTERS))


if __name__ == "__main__":
    main()
