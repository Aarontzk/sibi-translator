"""Export an MLP for the browser demo in web/. Owner: Azka.

Trains the same kind of MLP as train_model.py (same data, split and seed),
then writes its weights to web/public/model.json so web/src/sibi.js can run the
forward pass without Python. RandomForest is not exported: an MLP is a few small
matrices, a forest would be megabytes of JSON.

Usage:
  python export_web.py
"""
import base64
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from train_model import SEED, load_split

OUT_PATH = os.path.join("web", "public", "model.json")
MLP_PARAMS = {"hidden_layer_sizes": (128,), "alpha": 1e-4}


def main():
    Xtr, Xte, ytr, yte = load_split()

    mlp = MLPClassifier(max_iter=1000, random_state=SEED, **MLP_PARAMS).fit(Xtr, ytr)
    print(f"hold-out acc {accuracy_score(yte, mlp.predict(Xte)):.4f}")

    if mlp.activation != "relu" or mlp.out_activation_ != "softmax":
        raise SystemExit(f"web/src/sibi.js only runs relu+softmax, got {mlp.activation}/{mlp.out_activation_}")
    # int8 weights (one scale per layer) keep the file ~16 KB; check the accuracy cost
    layers, qcoefs = [], []
    for W, b in zip(mlp.coefs_, mlp.intercepts_):
        scale = float(np.abs(W).max()) / 127
        q = np.round(W / scale).astype(np.int8)
        qcoefs.append(q.astype(np.float64) * scale)
        layers.append({"rows": W.shape[0], "cols": W.shape[1], "scale": scale,
                       "W": base64.b64encode(q.tobytes()).decode(), "b": b.round(6).tolist()})
    full = mlp.predict(Xte)
    mlp.coefs_ = qcoefs
    agree = (mlp.predict(Xte) == full).mean()
    print(f"int8 model agrees with float model on {agree:.2%} of hold-out rows")
    model = {"labels": mlp.classes_.tolist(), "layers": layers}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"{len(model['labels'])} labels -> {OUT_PATH} ({os.path.getsize(OUT_PATH) // 1024} KB)")


if __name__ == "__main__":
    main()
