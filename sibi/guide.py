"""SIBI handshape guide: one picture per letter, cut from the alphabet poster.

The poster (web/public/abjad-sibi.png, 412x413) is the reference every recorder
copies, so the dataset matches what the web page shows. Boxes are (x0, y0, x1, y1)
in poster pixels; check them with ``python -m sibi.guide`` if the poster changes.
"""
import os

import cv2
import numpy as np

from sibi.features import LETTERS

GUIDE_PATH = os.path.join("web", "public", "abjad-sibi.png")

_ROWS = {  # letters: (top, bottom, x-centre of each letter)
    "ABCDEF": (24, 92, [55, 125, 192, 240, 300, 358]),
    "GHIKL": (98, 164, [50, 118, 182, 300, 358]),
    "MNOPQR": (172, 238, [48, 122, 183, 240, 300, 358]),
    "STUVWX": (240, 308, [48, 120, 182, 232, 293, 362]),
    "Y": (314, 384, [42]),
}
_HALF_W = {"H": 40}  # H is drawn sideways and wider than the others
GUIDE_BOXES = {
    L: (max(0, x - _HALF_W.get(L, 32)), top, x + _HALF_W.get(L, 32), bottom)
    for letters, (top, bottom, xs) in _ROWS.items()
    for L, x in zip(letters, xs)
}


def load_guide(path=GUIDE_PATH):
    """Return {letter: BGR image} cropped from the poster, or {} if it is missing."""
    poster = cv2.imread(path)
    if poster is None:
        print(f"Guide image {path} not found -- recording without the SIBI guide.")
        return {}
    return {L: poster[y0:y1, x0:x1].copy() for L, (x0, y0, x1, y1) in GUIDE_BOXES.items()}


def _demo(out="guide_check.png"):
    """Write every crop side by side so the boxes can be checked by eye."""
    guide = load_guide()
    assert sorted(guide) == sorted(LETTERS), sorted(set(LETTERS) ^ set(guide))
    tiles = [cv2.resize(guide[L], (96, 100)) for L in LETTERS]
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(out, sheet)
    print(f"guide OK -> {out}")


if __name__ == "__main__":
    _demo()
