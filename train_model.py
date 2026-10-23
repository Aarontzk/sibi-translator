"""Train Random Forest and MLP on landmarks.csv, compare, save the best.
Owner: Azka.

    python train_model.py                   # team data only
    python train_model.py --with-external   # + data/raw/external_train.csv (from convert_videos.py)
    python train_model.py --external-only   # external data only (no team recordings yet)

--with-external adds external rows to the TRAINING portion only; the hold-out
stays pure team data, so evaluate.py's team split is unchanged and comparable.
--external-only trains on external_train.csv and tests on external_test.csv,
keeping convert_videos.py's per-video split; GridSearch folds are grouped by
video too, so frames of one video never sit on both sides of a fold.

Loads data/raw/landmarks.csv, normalizes every row through
features.normalize_landmarks, runs a small GridSearch for RF and MLP, compares
accuracy and inference time on a stratified hold-out, prints an RF-vs-MLP table,
and saves the best model to models/sibi_model.joblib as a joblib dict
{"model", "labels"} (predict_proba is used downstream for confidence).
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from sibi.features import CSV_COORD_COLS, normalize_landmarks

DATA_PATH = os.path.join("data", "raw", "landmarks.csv")
EXT_TRAIN_PATH = os.path.join("data", "raw", "external_train.csv")
EXT_TEST_PATH = os.path.join("data", "raw", "external_test.csv")
MODEL_PATH = os.path.join("models", "sibi_model.joblib")
SEED = 42
# Trunojoyo signs Y with two hands (not SIBI one-hand Y); one-hand landmarks of it are noise
EXT_EXCLUDED_LABELS = frozenset({"Y"})


def load_xyg(path):
    """Return (X normalized 63-dim, y labels, person/video groups)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run collect_dataset.py / convert_videos.py first.")
    df = pd.read_csv(path)
    if path in (EXT_TRAIN_PATH, EXT_TEST_PATH):
        df = df[~df["label"].isin(EXT_EXCLUDED_LABELS)]
    if df.empty:
        raise SystemExit(f"{path} has no rows -- check how it was produced.")
    raw = df[CSV_COORD_COLS].to_numpy(dtype=np.float32)
    X = np.array([normalize_landmarks(r) for r in raw], dtype=np.float32)
    return X, df["label"].to_numpy(), df["person"].to_numpy()


def load_xy(path):
    """Return (X normalized 63-dim, y labels). Raw coords live in the CSV."""
    X, y, _ = load_xyg(path)
    return X, y


def mean_inference_ms(model, X):
    """Mean single-sample inference time in ms (batch timed, averaged)."""
    start = time.perf_counter()
    model.predict(X)
    return (time.perf_counter() - start) / len(X) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--with-external", action="store_true",
                      help=f"also train on {EXT_TRAIN_PATH}")
    mode.add_argument("--external-only", action="store_true",
                      help=f"train on {EXT_TRAIN_PATH}, test on {EXT_TEST_PATH}")
    args = ap.parse_args()

    cv, groups = 3, None
    if args.external_only:
        Xtr, ytr, groups = load_xyg(EXT_TRAIN_PATH)
        Xte, yte = load_xy(EXT_TEST_PATH)
        cv = GroupKFold(n_splits=3)  # folds split by video, not by frame
        print(f"External only: {len(ytr)} train / {len(yte)} test rows "
              f"({len(set(groups))} train videos)")
    else:
        X, y = load_xy(DATA_PATH)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        print(f"Team data: {len(ytr)} train / {len(yte)} hold-out rows")
        if args.with_external:
            Xext, yext = load_xy(EXT_TRAIN_PATH)
            missing = sorted(set(yext) - set(y))
            if missing:  # fine while the team is still recording
                print(f"Note: team hold-out has no samples for {' '.join(missing)}; "
                      f"check those with evaluate.py --external")
            Xtr = np.concatenate([Xtr, Xext])
            ytr = np.concatenate([ytr, yext])
            print(f"+ external: {len(yext)} rows -> {len(ytr)} train rows")

    candidates = {
        "RandomForest": (
            RandomForestClassifier(random_state=SEED),
            {"n_estimators": [200, 400], "max_depth": [None, 20]},
        ),
        "MLP": (
            MLPClassifier(max_iter=1000, random_state=SEED),
            {"hidden_layer_sizes": [(128,), (128, 64)], "alpha": [1e-4, 1e-3]},
        ),
    }

    results = {}
    for name, (est, grid) in candidates.items():
        gs = GridSearchCV(est, grid, cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr, groups=groups)
        best = gs.best_estimator_
        acc = accuracy_score(yte, best.predict(Xte))
        results[name] = {
            "model": best,
            "acc": acc,
            "infer_ms": mean_inference_ms(best, Xte),
            "params": gs.best_params_,
        }

    print(f"\n{'Model':<14}{'Test acc':>10}{'Infer ms/sample':>18}   Best params")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<14}{r['acc']:>10.4f}{r['infer_ms']:>18.3f}   {r['params']}")

    best_name = max(results, key=lambda n: results[n]["acc"])
    best_model = results[best_name]["model"]
    labels = best_model.classes_.tolist()

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": best_model, "labels": labels}, MODEL_PATH)
    print(f"\nBest: {best_name} (acc {results[best_name]['acc']:.4f}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
