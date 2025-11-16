#!/usr/bin/env python3
import time
import math
from pathlib import Path
import numpy as np


import cv2
from flask import Flask, Response

# ----------------------------
# Config
# ----------------------------

DRONE_ALT_M = 30.0          # assumed height for rough speed estimate
SPEED_LIMIT_MPH = 35.0      # demo speed limit
HFOV_DEG = 62.0             # Pi v2 approx horizontal FOV

CONF_THRESH = 0.5
NMS_THRESH = 0.4

VEHICLE_CLASS_IDS = {3, 4, 6, 8}  # COCO: car, motorcycle, bus, truck

app = Flask(__name__)

# ----------------------------
# Camera (Pi v2 via Argus)
# ----------------------------

def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

pipeline = gstreamer_pipeline(sensor_id=0)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("[ERR] Could not open CSI camera pipeline")
    cap = None
else:
    print("[INFO] CSI pipeline opened")

# ----------------------------
# DNN model (SSD MobileNet v3)
# ----------------------------

DEMO_DIR = Path(__file__).resolve().parent
V4_DIR = DEMO_DIR.parent
MODELS_DIR = V4_DIR / "models"

MODEL_PATH = str(MODELS_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pb")
CONFIG_PATH = str(MODELS_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

net = cv2.dnn_DetectionModel(MODEL_PATH, CONFIG_PATH)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

print("[INFO] Loaded SSD MobileNet v3")

# ----------------------------
# Simple tracking & speed
# ----------------------------

next_track_id = 1
tracks = {}  # track_id -> dict(center=(x,y), t, speed_mph)

HFOV_RAD = HFOV_DEG * math.pi / 180.0

def pixel_dx_to_speed_mph(dx_pixels, dt, frame_width):
    if dt <= 0:
        return 0.0
    pixel_to_rad = HFOV_RAD / frame_width
    vx_rad_per_s = dx_pixels / dt * pixel_to_rad
    speed_mps = abs(vx_rad_per_s * DRONE_ALT_M)
    return speed_mps * 2.23694  # m/s -> mph

def match_tracks(detections, frame_width):
    """
    detections: list of (box, cx, cy)
    Updates global 'tracks' with new centers & speeds.
    Returns list of (box, speed_mph).
    """
    global next_track_id, tracks

    now = time.time()
    used_track_ids = set()
    results = []

    for box, cx, cy in detections:
        best_id = None
        best_dist = 1e9

        for tid, info in tracks.items():
            if tid in used_track_ids:
                continue
            px, py = info["center"]
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        # if close enough, update existing track; else create new
        if best_id is not None and best_dist < 80:  # pixels
            info = tracks[best_id]
            dt = now - info["t"]
            dx = cx - info["center"][0]
            speed_mph = pixel_dx_to_speed_mph(dx, dt, frame_width)
            info["center"] = (cx, cy)
            info["t"] = now
            info["speed_mph"] = 0.7 * info.get("speed_mph", 0.0) + 0.3 * speed_mph
            used_track_ids.add(best_id)
            results.append((box, info["speed_mph"]))
        else:
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = {
                "center": (cx, cy),
                "t": now,
                "speed_mph": 0.0,
            }
            used_track_ids.add(tid)
            results.append((box, 0.0))

    # prune stale tracks
    to_delete = [tid for tid, info in tracks.items() if now - info["t"] > 2.0]
    for tid in to_delete:
        del tracks[tid]

    return results

# ----------------------------
# Streaming generator
# ----------------------------

def generate_frames():
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[STREAM] Camera error or end of stream.")
            break

        h, w = frame.shape[:2]

        class_ids, confidences, boxes = net.detect(frame, CONF_THRESH, NMS_THRESH)

        detections = []
        if class_ids is not None and len(class_ids) > 0:
            class_ids = np.array(class_ids).flatten()
            confidences = np.array(confidences).flatten()

            for class_id, conf, box in zip(class_ids, confidences, boxes):
                if class_id not in VEHICLE_CLASS_IDS:
                    continue
                x, y, bw, bh = box
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                detections.append((box, cx, cy))


        # match detections to tracks and estimate speed
        boxes_with_speed = match_tracks(detections, w)

        # draw boxes
        for box, speed_mph in boxes_with_speed:
            x, y, bw, bh = box
            over = speed_mph > SPEED_LIMIT_MPH
            color = (0, 0, 255) if over else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            label = f"{speed_mph:4.1f} mph"
            if over:
                label += f" > {SPEED_LIMIT_MPH}"
            cv2.putText(frame, label, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # encode and yield
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# ----------------------------
# Flask routes
# ----------------------------

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/")
def index():
    return (
        "<html><body>"
        "<h2>Flyby Multi-Car Speed Demo</h2>"
        "<img src='/video' />"
        "</body></html>"
    )

if __name__ == "__main__":
    print("[INFO] Starting multi-car speed stream on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
