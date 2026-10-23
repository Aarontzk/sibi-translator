"""Model evaluation. Owner: Rafi.

    python evaluate.py              # team hold-out (same split as train_model.py)
    python evaluate.py --external   # data/raw/external_test.csv (videos never trained on)

Prints a per-letter classification report, saves a confusion-matrix heatmap to
reports/ (confusion_matrix.png or confusion_matrix_external.png), and lists the 5
most-confused letter pairs.
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sibi.features import normalize_landmarks  # noqa: F401 (keeps single-norm contract explicit)
from train_model import DATA_PATH, EXT_TEST_PATH, MODEL_PATH, SEED, load_xy

DOCS_DIR = "reports"


def top_confused_pairs(cm, labels, k=5):
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def save_heatmap(cm, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (hold-out)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", action="store_true",
                    help=f"test on {EXT_TEST_PATH} instead of the team hold-out")
    args = ap.parse_args()

    if args.external:
        Xte, yte = load_xy(EXT_TEST_PATH)  # already split per video, use whole file
        name = "confusion_matrix_external.png"
    else:
        X, y = load_xy(DATA_PATH)
        _, Xte, _, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        name = "confusion_matrix.png"
    print(f"Test set: {'external' if args.external else 'team hold-out'}, {len(yte)} rows")

    bundle = joblib.load(MODEL_PATH)
    model, labels = bundle["model"], bundle["labels"]

    pred = model.predict(Xte)
    print(classification_report(yte, pred, labels=labels, zero_division=0))

    cm = confusion_matrix(yte, pred, labels=labels)
    os.makedirs(DOCS_DIR, exist_ok=True)
    out = os.path.join(DOCS_DIR, name)
    save_heatmap(cm, labels, out)
    print(f"Saved {out}")

    print("\nTop 5 confused pairs (true -> predicted, count):")
    for count, t, p in top_confused_pairs(cm, labels):
        print(f"  {t} -> {p}: {count}")


if __name__ == "__main__":
    main()
