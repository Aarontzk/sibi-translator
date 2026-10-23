// Camera + MediaPipe loop. Calls onFrame with each reading; draws onto the canvas.
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { Stabilizer, decodeModel, mlpPredict, normalizeLandmarks } from "./sibi.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
// Python ran ~12-15 FPS; inferring at the same rate keeps the Stabilizer timing identical.
const INFER_INTERVAL_MS = 1000 / 15;
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const LINE_COLOR = "#0B64DD"; // EUI Borealis primary
const POINT_COLOR = "#FFFFFF";

async function loadModel() {
  const r = await fetch("/model.json");
  if (!r.ok) throw new Error(`model.json: HTTP ${r.status}`);
  return decodeModel(await r.json());
}

function draw(ctx, video, landmarks) {
  const { width: w, height: h } = ctx.canvas;
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1); // mirror so it feels like a mirror
  ctx.drawImage(video, 0, 0, w, h);
  if (landmarks) {
    ctx.strokeStyle = LINE_COLOR; ctx.lineWidth = 3;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
      ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      ctx.stroke();
    }
    ctx.fillStyle = POINT_COLOR;
    for (const p of landmarks) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 4, 0, 7); ctx.fill(); }
  }
  ctx.restore();
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {(r:{letter:string|null, conf:number, progress:number, hasHand:boolean, emitted:string|null}) => void} onFrame
 * @param {(stage:string) => void} onStage
 * @returns {Promise<() => void>} stop function
 */
export async function startDetector(canvas, onFrame, onStage) {
  onStage("Memuat model…");
  const [model, fileset] = await Promise.all([loadModel(), FilesetResolver.forVisionTasks(WASM_URL)]);
  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO", numHands: 1,
    minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6,
  });
  onStage("Meminta izin kamera…");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" }, audio: false,
  });
  const video = Object.assign(document.createElement("video"), { playsInline: true, muted: true, srcObject: stream });
  await video.play();

  const ctx = canvas.getContext("2d");
  const stab = new Stabilizer();
  let raf = 0, lastInfer = 0, landmarks = null;
  const tick = (now) => {
    if (video.readyState >= 2 && now - lastInfer >= INFER_INTERVAL_MS) {
      lastInfer = now;
      landmarks = landmarker.detectForVideo(video, now).landmarks?.[0] ?? null;
      const aspect = video.videoWidth / video.videoHeight;
      const { letter, conf } = landmarks ? mlpPredict(model, normalizeLandmarks(landmarks, aspect)) : { letter: null, conf: 0 };
      const emitted = stab.update(letter, conf);
      onFrame({ letter, conf, progress: stab.progress, hasHand: !!landmarks, emitted });
    }
    draw(ctx, video, landmarks);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return () => {
    cancelAnimationFrame(raf);
    stream.getTracks().forEach((t) => t.stop());
    landmarker.close();
  };
}
