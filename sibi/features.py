"""Feature extraction for SIBI fingerspelling. Owner: Azka.

Single source of landmark normalization. ``normalize_landmarks`` is the ONLY
place normalization happens -- this is the interface contract shared with the
whole team (collect_dataset stores raw coords; train/evaluate/app all normalize
through here).
"""
import numpy as np

# 24 static SIBI letters: A-Y without J (J and Z need motion, out of scope).
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 letters
N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3  # 63

# data/raw/landmarks.csv column contract (owner: Farel, consumers: Azka, Rafi).
CSV_COORD_COLS = [f"{ax}{i}" for i in range(N_LANDMARKS) for ax in ("x", "y", "z")]
CSV_COLUMNS = ["label", "person", "hand"] + CSV_COORD_COLS


def landmarks_to_array(hand_landmarks):
    """MediaPipe NormalizedLandmarkList -> (21, 3) float array of raw x, y, z."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )


def normalize_landmarks(lm):
    """Normalize 21 hand landmarks into a 63-value feature vector.

    Input: array-like of 21 points (x, y, z) -- shape (21, 3) or flat (63,),
    or a MediaPipe landmark list (anything with ``.landmark``).
    Output: numpy array of 63 values, translated so landmark 0 (wrist) sits at
    the origin and scaled by the maximum distance from it. Result is invariant
    to hand position and scale in the frame. Single agreed normalization point.
    """
    if hasattr(lm, "landmark"):
        pts = landmarks_to_array(lm)
    else:
        pts = np.asarray(lm, dtype=np.float32).reshape(N_LANDMARKS, 3)

    pts = pts - pts[0]                       # relative to wrist (landmark 0)
    max_dist = float(np.linalg.norm(pts, axis=1).max())
    if max_dist > 0:
        pts = pts / max_dist                 # scale by farthest point
    return pts.reshape(-1).astype(np.float32)  # 63 values


def _demo():
    rng = np.random.default_rng(0)
    raw = rng.random((21, 3), dtype=np.float32)
    f = normalize_landmarks(raw)
    assert f.shape == (N_FEATURES,), f.shape
    assert np.allclose(f[:3], 0.0)                              # wrist -> origin
    assert np.allclose(f, normalize_landmarks(raw * 5.0), atol=1e-5)   # scale inv.
    assert np.allclose(f, normalize_landmarks(raw + 10.0), atol=1e-5)  # transl inv.
    assert np.allclose(f, normalize_landmarks(raw.reshape(-1)))        # flat input
    assert len(LETTERS) == 24 and "J" not in LETTERS and "Z" not in LETTERS
    assert len(CSV_COORD_COLS) == N_FEATURES
    print("features OK")


if __name__ == "__main__":
    _demo()
