"""Train Random Forest and MLP on SIBI landmarks, compare, save the best.
Owner: Azka.

    python train_model.py

Data (both optional, at least one needed):
- data/raw/landmarks.csv     team recordings (collect_dataset.py, SIBI guide image)
- data/raw/sibi_mendeley.csv Mendeley SIBI photos (convert_images.py)

Every row is normalized through features.normalize_landmarks. A small GridSearch
picks RF and MLP settings, both are compared on a hold-out (accuracy and
inference time), and the best is saved to models/sibi_model.joblib as a joblib
dict {"model", "labels"} (predict_proba is used downstream for confidence).

Hold-out (load_split, shared by evaluate.py and export_web.py):
- Mendeley rows: the dataset authors' test/ split.
- Team rows: with 3+ recorders whole people are held out, so the score says how
  well the model reads a hand it has never seen; otherwise a per-frame split.
Each person records ~100 near-identical frames per letter, so (person, letter)
groups are capped at MAX_PER_PERSON_LETTER rows to keep one person or one letter
from outweighing the rest.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier

from sibi.features import CSV_COORD_COLS, LETTERS, normalize_landmarks

DATA_PATH = os.path.join("data", "raw", "landmarks.csv")
MENDELEY_PATH = os.path.join("data", "raw", "sibi_mendeley.csv")
MODEL_PATH = os.path.join("models", "sibi_model.joblib")
SEED = 42
MIN_PEOPLE_FOR_GROUP_SPLIT = 3
MAX_PER_PERSON_LETTER = 150


def read_rows():
    """All available CSVs as one DataFrame, (person, letter) groups capped."""
    frames = [pd.read_csv(p) for p in (DATA_PATH, MENDELEY_PATH) if os.path.exists(p)]
    if not frames:
        raise SystemExit(f"No data: record with collect_dataset.py ({DATA_PATH}) "
                         f"or run convert_images.py ({MENDELEY_PATH}).")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise SystemExit("Data files have no rows.")
    df = df.sample(frac=1, random_state=SEED)  # shuffle, then keep the first N per group
    df = df[df.groupby(["person", "label"]).cumcount() < MAX_PER_PERSON_LETTER]
    return df.sort_index().reset_index(drop=True)


def features(df):
    raw = df[CSV_COORD_COLS].to_numpy(dtype=np.float32)
    return np.array([normalize_landmarks(r) for r in raw], dtype=np.float32)


def check_letters(y):
    """Stop if there is nothing to classify; warn about letters with no data."""
    present = set(y)
    if len(present) < 2:
        raise SystemExit(f"Only {sorted(present)} available -- need at least 2 letters to train.")
    missing = [L for L in LETTERS if L not in present]
    if missing:
        print(f"Warning: no samples for {' '.join(missing)} -- the model cannot output these.")


def team_test_mask(df):
    """Hold-out mask for team rows (see module docstring)."""
    mask = np.zeros(len(df), dtype=bool)
    if df.empty:
        return mask
    groups = df["person"].to_numpy()
    if len(set(groups)) >= MIN_PEOPLE_FOR_GROUP_SPLIT:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        _, te = next(gss.split(df, groups=groups))
    else:
        print(f"Note: fewer than {MIN_PEOPLE_FOR_GROUP_SPLIT} team recorders -- per-frame split "
              f"for team data, its hold-out accuracy will look higher than it really is.")
        y = df["label"].to_numpy()
        strat = y if min(pd.Series(y).value_counts()) >= 2 else None
        _, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=strat, random_state=SEED)
    mask[te] = True
    return mask


def load_split():
    """Return (Xtr, Xte, ytr, yte) over all data, split as in the module docstring."""
    df = read_rows()
    is_mendeley = df["person"].str.startswith("mendeley/").to_numpy()
    test = np.zeros(len(df), dtype=bool)
    test[is_mendeley] = (df.loc[is_mendeley, "person"] == "mendeley/test").to_numpy()
    test[~is_mendeley] = team_test_mask(df[~is_mendeley].reset_index(drop=True))
    X, y = features(df), df["label"].to_numpy()
    check_letters(y[~test])
    print(f"Rows: {int((~is_mendeley).sum())} team ({df.loc[~is_mendeley, 'person'].nunique()} people), "
          f"{int(is_mendeley.sum())} Mendeley -> {int((~test).sum())} train / {int(test.sum())} hold-out")
    return X[~test], X[test], y[~test], y[test]


def mean_inference_ms(model, X):
    """Mean single-sample inference time in ms (batch timed, averaged)."""
    start = time.perf_counter()
    model.predict(X)
    return (time.perf_counter() - start) / len(X) * 1000.0


def main():
    Xtr, Xte, ytr, yte = load_split()

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
        gs = GridSearchCV(est, grid, cv=3, n_jobs=-1)
        gs.fit(Xtr, ytr)
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
