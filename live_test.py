"""Browser UI for live testing. Owner: Rafi.

    python live_test.py [--camera 0] [--port 8000]
    -> open http://127.0.0.1:8000

Runs the exact app_realtime.py pipeline (camera -> flip -> MediaPipe -> normalize ->
model -> Stabilizer) in a background thread and serves:
    GET  /          live_test.html
    GET  /stream    MJPEG video with landmarks
    GET  /state     JSON: letter, conf, progress, fps, hand, seq/last (commits)
    POST /reset     fresh Stabilizer (called at the start of every trial)
    POST /results   save a finished test to reports/live_test/<tester>_<scenario>_<time>.csv

Stdlib HTTP server only, bound to 127.0.0.1 (not reachable from the network).
"""
import argparse
import csv
import json
import os
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mediapipe as mp

from app_realtime import MODEL_PATH, load_model, predict
from sibi.features import LETTERS
from sibi.stabilizer import Stabilizer

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_test.html")
RESULTS_DIR = os.path.join("reports", "live_test")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_TRIALS = 2000
MAX_BODY = 1_000_000


class Pipeline(threading.Thread):
    def __init__(self, camera):
        super().__init__(daemon=True)
        self.camera = camera
        self.lock = threading.Lock()
        self.jpeg = None
        self.stab = Stabilizer(n_frames=8, min_conf=0.7, cooldown=12)
        self.state = {"letter": None, "conf": 0.0, "progress": 0.0, "fps": 0.0,
                      "hand": False, "seq": 0, "last": None, "error": None}

    def reset(self):
        with self.lock:
            s = self.stab
            self.stab = Stabilizer(s.n_frames, s.min_conf, s.cooldown)

    def snapshot(self):
        with self.lock:
            return dict(self.state), self.jpeg

    def _fail(self, msg):
        print("ERROR:", msg)
        with self.lock:
            self.state["error"] = msg

    def run(self):
        try:
            model, labels = load_model(MODEL_PATH)
        except SystemExit as e:
            self._fail(str(e))
            return
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            self._fail(f"Cannot open camera {self.camera}")
            return
        hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        draw = mp.solutions.drawing_utils
        prev, fps = time.perf_counter(), 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    self._fail("Frame grab failed")
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand, letter, conf = None, None, 0.0
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    letter, conf = predict(model, labels, hand)
                    draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
                prev = now
                encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

                with self.lock:
                    committed = self.stab.update(letter, conf)
                    self.state.update(letter=letter, conf=round(conf, 3),
                                      progress=round(self.stab.progress, 3),
                                      fps=round(fps, 1), hand=hand is not None)
                    if committed is not None:
                        self.state["seq"] += 1
                        self.state["last"] = committed
                    if encoded:
                        self.jpeg = buf.tobytes()
        finally:
            cap.release()
            hands.close()


def save_results(payload):
    """Validate a POSTed test run and write it as CSV. Returns the file path."""
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")
    tester, scenario = str(payload.get("tester", "")), str(payload.get("scenario", ""))
    if not NAME_RE.match(tester) or not NAME_RE.match(scenario):
        raise ValueError("tester/scenario: 1-32 chars, letters, digits, _ or -")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not 0 < len(trials) <= MAX_TRIALS:
        raise ValueError(f"trials must be a list of 1-{MAX_TRIALS} items")

    rows = []
    for t in trials:
        if not isinstance(t, dict):
            raise ValueError("each trial must be an object")
        target, got, ms = t.get("target"), t.get("got") or "", t.get("ms")
        if target not in LETTERS or (got and got not in LETTERS) \
                or not isinstance(ms, (int, float)) or ms < 0:
            raise ValueError(f"invalid trial: {t}")
        rows.append([tester, scenario, target, got, int(got == target), int(ms)])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{tester}_{scenario}_{datetime.now():%Y%m%d-%H%M%S}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tester", "scenario", "target", "predicted", "correct", "ms"])
        w.writerows(rows)
    return path


def make_handler(pipe):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                with open(PAGE_PATH, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state":
                state, _ = pipe.snapshot()
                self._json(200, state)
            elif self.path == "/stream":
                self._stream()
            else:
                self._json(404, {"error": "not found"})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    _, jpg = pipe.snapshot()
                    if jpg is not None and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # browser closed the stream

        def do_POST(self):
            if self.path == "/reset":
                pipe.reset()
                self._json(200, {"ok": True})
            elif self.path == "/results":
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY:
                    self._json(413, {"error": "body missing or too large"})
                    return
                try:
                    path = save_results(json.loads(self.rfile.read(length)))
                except (ValueError, json.JSONDecodeError) as e:
                    self._json(400, {"error": str(e)})
                    return
                print("saved", path)
                self._json(200, {"ok": True, "path": path})
            else:
                self._json(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    pipe = Pipeline(args.camera)
    pipe.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(pipe))
    server.daemon_threads = True
    print(f"Live test UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
