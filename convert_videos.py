"""Convert an external sign-language VIDEO dataset into landmark CSVs.

    python convert_videos.py VIDEO_DIR --source trunojoyo [--every 5] [--hand right]

Output (same column contract as data/raw/landmarks.csv, raw coords):
    data/raw/external_train.csv   ~80% of the videos of each letter
    data/raw/external_test.csv    ~20% of the videos of each letter

Kept separate from data/raw/landmarks.csv on purpose: the team's own recordings
stay the primary dataset; external data is supplementary and must be cited.

Rules:
- Frames are flipped horizontally exactly like app_realtime.py / collect_dataset.py, so
  landmarks are not mirrored relative to live webcam input.
- The split is PER VIDEO, never per frame: frames of one video are nearly
  identical, so a per-frame split would leak and inflate accuracy. The
  ``person`` column holds "<source>/<video name>" to keep each video traceable.
- Letters outside the 24-letter scope (J, Z) are skipped.
- Label = parent folder name if it is a single letter (A/, B/, ...), otherwise
  the first character of the file name (A_01.mov, b3.mp4, ...).
"""
import argparse
import csv
import os
import random
from collections import defaultdict

import cv2
import mediapipe as mp

from sibi.features import CSV_COLUMNS, LETTERS, landmarks_to_array

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".m4v")
SEED = 42


def infer_label(path):
    parent = os.path.basename(os.path.dirname(path))
    if len(parent) == 1 and parent.isalpha():
        return parent.upper()
    return os.path.basename(path)[0].upper()


def find_videos(root):
    found = []
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(VIDEO_EXT):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def split_by_video(videos_by_label, test_frac=0.2, seed=SEED):
    """Per-letter split of VIDEOS. Letters with >=2 videos get >=1 test video."""
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(videos_by_label):
        vids = sorted(videos_by_label[label])
        rng.shuffle(vids)
        n_test = round(len(vids) * test_frac)
        if len(vids) >= 2:
            n_test = max(n_test, 1)
        test += vids[:n_test]
        train += vids[n_test:]
    return train, test


def extract_rows(path, label, person, hand_name, hands, every, start=0.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ! cannot open {path}, skipped")
        return [], 0
    first = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * start)  # skip the hand raising up
    rows, i, sampled = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= first and (i - first) % every == 0:
            sampled += 1
            frame = cv2.flip(frame, 1)  # same as app_realtime.py
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                pts = landmarks_to_array(result.multi_hand_landmarks[0]).reshape(-1)
                rows.append([label, person, hand_name, *pts.tolist()])
        i += 1
    cap.release()
    return rows, sampled


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--source", required=True, help="short dataset name, e.g. trunojoyo")
    ap.add_argument("--every", type=int, default=5, help="use every Nth frame")
    ap.add_argument("--start", type=float, default=0.0,
                    help="skip this fraction of each video first (e.g. 0.5 when "
                         "videos start with the hand still rising)")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--out-dir", default=os.path.join("data", "raw"))
    args = ap.parse_args()
    if args.every < 1 or not 0.0 <= args.start < 1.0:
        raise SystemExit("--every must be >= 1 and --start in [0, 1)")

    videos_by_label = defaultdict(list)
    skipped = defaultdict(int)
    for path in find_videos(args.video_dir):
        label = infer_label(path)
        if label in LETTERS:
            videos_by_label[label].append(path)
        else:
            skipped[label] += 1
    if not videos_by_label:
        raise SystemExit(f"No usable videos found under {args.video_dir}")
    if skipped:
        print("Skipped (out of scope / unknown label):", dict(skipped))

    train_vids, test_vids = split_by_video(videos_by_label)
    test_set = set(test_vids)

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6
    )
    rows = {"train": [], "test": []}
    no_hand = 0
    sampled_total = 0
    for path in train_vids + test_vids:
        label = infer_label(path)
        person = f"{args.source}/{os.path.splitext(os.path.basename(path))[0]}"
        got, sampled = extract_rows(path, label, person, args.hand, hands,
                                    args.every, args.start)
        sampled_total += sampled
        no_hand += sampled - len(got)
        rows["test" if path in test_set else "train"] += got
        print(f"  {label} {os.path.basename(path)}: {len(got)}/{sampled} frames with hand")
    hands.close()

    os.makedirs(args.out_dir, exist_ok=True)
    for part in ("train", "test"):
        out = os.path.join(args.out_dir, f"external_{part}.csv")
        write_csv(out, rows[part])
        print(f"{out}: {len(rows[part])} rows")
    print(f"Videos: {len(train_vids)} train / {len(test_vids)} test. "
          f"Frames without a hand dropped: {no_hand}/{sampled_total}")
    missing = [L for L in LETTERS if L not in videos_by_label]
    if missing:
        print("Letters with no videos:", " ".join(missing))


if __name__ == "__main__":
    main()
