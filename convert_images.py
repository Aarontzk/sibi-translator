"""Convert the Mendeley SIBI photo dataset into landmark rows. Owner: Farel.

    python convert_images.py data/images/sibi_mendeley

Input: the dataset's sibi_alphabet_mono_bg_no_augmentation/images folder, with its
own train/ val/ test/ subfolders. The letter is the first character of each file
name (A_33_jpg.rf....jpg). Output: data/raw/sibi_mendeley.csv in the same columns
as landmarks.csv; ``person`` is "mendeley/<split>" so train_model.py can keep the
dataset authors' test split as hold-out.

- Every photo is flipped like the webcam frames in collect_dataset.py.
- Left-hand photos are mirrored into right-hand landmarks (x -> 1 - x), because
  the team signs with the right hand.
- Photos where MediaPipe finds no hand are skipped and counted.

Dataset (cite in References, CC BY 4.0): Andriyanto, T. "Labeled Official
Indonesian Sign Language (SIBI) Alphabet". Mendeley Data, v1, 2024.
doi:10.17632/yhsgmynccf.1
"""
import argparse
import csv
import os
from collections import Counter

import cv2
import mediapipe as mp

from sibi.features import CSV_COLUMNS, LETTERS, landmarks_to_array

OUT_PATH = os.path.join("data", "raw", "sibi_mendeley.csv")
SPLITS = ("train", "val", "test")
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def image_landmarks(hands, path):
    """Return ((21, 3) right-hand landmarks, detected side) for one photo, or None."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.flip(img, 1)  # same view as the flipped webcam frame
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None
    pts = landmarks_to_array(result.multi_hand_landmarks[0])
    side = result.multi_handedness[0].classification[0].label
    if side == "Left":
        pts[:, 0] = 1.0 - pts[:, 0]  # mirror a left hand into a right hand
    return pts, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", help="folder containing train/ val/ test/")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=0.5)
    stats, sides = Counter(), Counter()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for split in SPLITS:
            folder = os.path.join(args.image_dir, split)
            if not os.path.isdir(folder):
                raise SystemExit(f"{folder} not found -- pass the dataset's images/ folder")
            for name in sorted(os.listdir(folder)):
                letter = name[:1].upper()
                if not name.lower().endswith(IMAGE_EXT) or letter not in LETTERS:
                    continue
                found = image_landmarks(hands, os.path.join(folder, name))
                if found is None:
                    stats[f"{split} no hand"] += 1
                    continue
                pts, side = found
                sides[side] += 1
                stats[split] += 1
                writer.writerow([letter, f"mendeley/{split}", "right", *pts.reshape(-1).tolist()])
    hands.close()
    print(f"-> {args.out}: {dict(stats)}  handedness before mirroring: {dict(sides)}")


if __name__ == "__main__":
    main()
