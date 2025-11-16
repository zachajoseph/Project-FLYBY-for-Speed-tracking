#!/usr/bin/env python3
import time
import math

import cv2
from flask import Flask, Response
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------

DRONE_ALT_M = 30.0          # reserved for future geometry / speed
CONF_THRESH = 0.45          # YOLO confidence threshold
NMS_THRESH = 0.45           # IoU threshold

# COCO indices for YOLOv8:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

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
# YOLOv8s model on CUDA FP16
# ----------------------------

print("[INFO] Loading YOLOv8s (COCO)...")
model = YOLO("yolov8n.pt")
print("[INFO] YOLOv8s loaded")


# ----------------------------
# Simple temporal tracking (A) + COM
# ----------------------------

tracks = {}            # track_id -> dict(cx, cy, x1, y1, x2, y2, hits, last_t, conf, cls)
next_track_id = 1
MAX_MATCH_DIST = 80.0  # pixels
MIN_HITS_TO_SHOW = 2   # must be seen in >= this many frames
MAX_AGE = 0.5          # seconds before track is dropped


def update_tracks(detections):
    """
    detections: list of (x1, y1, x2, y2, cx, cy, conf, cls)
    returns: list of (x1, y1, x2, y2, cx, cy, conf, cls) for stable tracks
    """
    global tracks, next_track_id
    now = time.time()

    # mark all tracks as unused this frame
    for tid in tracks:
        tracks[tid]["used"] = False

    # associate detections to existing tracks by nearest center
    for x1, y1, x2, y2, cx, cy, conf, cls in detections:
        best_id = None
        best_dist = 1e9
        for tid, info in tracks.items():
            dx = cx - info["cx"]
            dy = cy - info["cy"]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None and best_dist < MAX_MATCH_DIST:
            info = tracks[best_id]
            info["cx"] = cx
            info["cy"] = cy
            info["x1"] = x1
            info["y1"] = y1
            info["x2"] = x2
            info["y2"] = y2
            info["hits"] += 1
            info["conf"] = conf
            info["cls"] = cls
            info["last_t"] = now
            info["used"] = True
        else:
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = {
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "hits": 1,
                "conf": conf,
                "cls": cls,
                "last_t": now,
                "used": True,
            }

    # drop stale tracks
    dead = [tid for tid, info in tracks.items() if now - info["last_t"] > MAX_AGE]
    for tid in dead:
        del tracks[tid]

    # collect stable tracks
    active = []
    for info in tracks.values():
        if info["hits"] >= MIN_HITS_TO_SHOW:
            active.append(
                (
                    info["x1"], info["y1"], info["x2"], info["y2"],
                    info["cx"], info["cy"], info["conf"], info["cls"],
                )
            )
    return active

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

        # Run YOLOv8 inference (device 0 = first CUDA GPU)
        results = model(
            frame,
            imgsz=768,
            conf=CONF_THRESH,
            iou=NMS_THRESH,
            device=0,      # use CUDA:0 if available
            verbose=False,
        )[0]

        # Geometric filters
        min_area = 0.0005 * w * h   # allow smaller cars
        max_area = 0.4 * w * h
        min_aspect = 0.3
        max_aspect = 3.5

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            aspect = bh / max(bw, 1)

            if not (min_area <= area <= max_area):
                continue
            if not (min_aspect <= aspect <= max_aspect):
                continue

            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0

            detections.append((x1, y1, x2, y2, cx, cy, conf, cls))

        # temporal smoothing + center-of-mass
        tracks_to_draw = update_tracks(detections)

        # draw stable tracks
        for x1, y1, x2, y2, cx, cy, conf, cls in tracks_to_draw:
            color = (0, 255, 0)  # green boxes for now

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # COM marker
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

            cls_name = model.names.get(cls, "veh")
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode and stream
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
        "<h2>Flyby YOLOv8s Vehicle Stream (tracked)</h2>"
        "<img src='/video' />"
        "</body></html>"
    )

if __name__ == "__main__":
    print("[INFO] Starting YOLOv8s vehicle stream on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
