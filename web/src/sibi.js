// Browser port of the Python pipeline: features.normalize_landmarks -> MLP -> Stabilizer.
// Keep in sync with features.py, stabilizer.py and export_web.py.

export const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

export const STABILIZER_DEFAULTS = { nFrames: 8, minConf: 0.7, cooldown: 12 };

/** Same rules as stabilizer.Stabilizer: emit once per held gesture. */
export class Stabilizer {
  constructor({ nFrames, minConf, cooldown } = STABILIZER_DEFAULTS) {
    Object.assign(this, { nFrames, minConf, cooldown });
    this.current = null; this.count = 0; this.cooldownLeft = 0; this.locked = null; this.progress = 0;
  }

  /** @returns {string|null} a newly committed letter */
  update(letter, conf) {
    if (this.cooldownLeft > 0) this.cooldownLeft -= 1;
    if (letter === null || conf < this.minConf) {
      this.current = null; this.count = 0; this.locked = null; this.progress = 0;
      return null;
    }
    if (letter === this.current) this.count += 1;
    else { this.current = letter; this.count = 1; this.locked = null; }
    if (letter === this.locked) { this.progress = 0; return null; }
    this.progress = Math.min(this.count / this.nFrames, 1);
    if (this.count >= this.nFrames && this.cooldownLeft === 0) {
      this.cooldownLeft = this.cooldown; this.count = 0; this.locked = letter; this.progress = 0;
      return letter;
    }
    return null;
  }
}

/**
 * Wrist at origin, scaled by farthest point. x is mirrored because the
 * Python pipeline flips every frame before MediaPipe sees it.
 * @param {{x:number,y:number,z:number}[]} lms 21 MediaPipe landmarks
 * @returns {number[]} 63 features
 */
export function normalizeLandmarks(lms) {
  const pts = lms.map((p) => [1 - p.x, p.y, p.z]);
  const [wx, wy, wz] = pts[0];
  const rel = pts.map(([x, y, z]) => [x - wx, y - wy, z - wz]);
  const maxDist = Math.max(...rel.map(([x, y, z]) => Math.hypot(x, y, z)));
  const scale = maxDist > 0 ? maxDist : 1;
  return rel.flat().map((v) => v / scale);
}

/** export_web.py stores each weight matrix as base64 int8 (row-major) plus one scale. */
export function decodeModel(raw) {
  const layers = raw.layers.map(({ rows, cols, scale, W, b }) => {
    const q = new Int8Array(Uint8Array.from(atob(W), (ch) => ch.charCodeAt(0)).buffer);
    if (q.length !== rows * cols) throw new Error("model.json rusak: ukuran bobot tidak cocok");
    return { cols, W: Float32Array.from(q, (v) => v * scale), b };
  });
  return { labels: raw.labels, layers };
}

/** ReLU hidden layers, softmax output. @returns {{letter:string, conf:number}} */
export function mlpPredict(model, x) {
  let h = x;
  model.layers.forEach(({ cols, W, b }, i) => {
    const out = b.slice();
    for (let r = 0; r < h.length; r++) {
      const v = h[r], off = r * cols;
      if (v !== 0) for (let c = 0; c < cols; c++) out[c] += v * W[off + c];
    }
    h = i < model.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const m = Math.max(...h);
  const e = h.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, v) => a + v, 0);
  let best = 0;
  for (let i = 1; i < e.length; i++) if (e[i] > e[best]) best = i;
  return { letter: model.labels[best], conf: e[best] / sum };
}
