"""Letter stabilizer. Owner: Gregorius.

Emit a letter only when the same prediction holds for ``n_frames`` consecutive
frames above ``min_conf``, then enforce a ``cooldown`` before the next emit.
A held gesture produces exactly one letter: the same letter is locked until the
gesture is released (hand gone, low confidence, or a different letter). Double
letters ("ANNA") = hold, release, hold again.

Interface contract with ui.py (Sahril): ``update(letter, conf)`` returns a new
letter or None; ``progress`` (0..1) drives the hold indicator.
"""


class Stabilizer:
    def __init__(self, n_frames=8, min_conf=0.7, cooldown=12):
        self.n_frames = n_frames
        self.min_conf = min_conf
        self.cooldown = cooldown
        self._current = None
        self._count = 0
        self._cooldown_left = 0
        self._locked = None  # last emitted letter, blocked until released
        self.progress = 0.0

    def update(self, letter, conf):
        """Feed one frame's (letter, conf). Return a newly committed letter or None."""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # No hand / low confidence: reset the hold.
        if letter is None or conf < self.min_conf:
            self._current = None
            self._count = 0
            self._locked = None
            self.progress = 0.0
            return None

        if letter == self._current:
            self._count += 1
        else:
            self._current = letter
            self._count = 1
            self._locked = None  # switched gesture -> release the lock

        if letter == self._locked:
            self.progress = 0.0
            return None

        self.progress = min(self._count / self.n_frames, 1.0)

        if self._count >= self.n_frames and self._cooldown_left == 0:
            self._cooldown_left = self.cooldown
            self._count = 0
            self._locked = letter
            self.progress = 0.0
            return letter
        return None


def _demo():
    s = Stabilizer(n_frames=3, min_conf=0.5, cooldown=2)
    assert [s.update("A", 0.9) for _ in range(3)] == [None, None, "A"]
    # holding the same gesture never repeats, even after cooldown
    assert all(s.update("A", 0.9) is None for _ in range(30))
    assert s.update(None, 0.0) is None     # release (hand gone)
    assert [s.update("A", 0.9) for _ in range(3)] == [None, None, "A"]  # "AA"
    # switching letter releases the lock; cooldown still gates the next emit
    s.update(None, 0.0)
    out = [s.update("B", 0.9) for _ in range(3)]
    assert out == [None, None, "B"], out
    s2 = Stabilizer(n_frames=2, min_conf=0.6)
    assert s2.update("B", 0.3) is None and s2.progress == 0.0  # low conf resets
    assert s2.update("B", 0.9) is None     # different gesture needs the hold
    print("stabilizer OK")


if __name__ == "__main__":
    _demo()
