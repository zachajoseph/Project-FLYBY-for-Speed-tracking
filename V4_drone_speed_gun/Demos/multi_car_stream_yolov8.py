#!/usr/bin/env python3
import cv2
from flask import Flask, Response
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------

DRONE_ALT_M = 30.0          # reserved for future geometry / speed (unused for now)
CONF_THRESH = 0.45          # YOLO confidence threshold
NMS_THRESH = 0.45           # IoU threshold

# COCO indices for YOLOv8:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

RUN_EVERY_N_FRAMES = 2      # minimal smoothing: run YOLO on every 2nd frame

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
# YOLOv8n model
# ----------------------------

print("[INFO] Loading YOLOv8n (COCO)...")
model = YOLO("yolov8n.pt")
print("[INFO] YOLOv8n loaded")

# ----------------------------
# Streaming generator
# ----------------------------

frame_idx = 0
last_boxes = []  # list of (x1, y1, x2, y2, cls, conf)

def generate_frames():
    global frame_idx, last_boxes
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[STREAM] Camera error or end of stream.")
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        run_yolo = (frame_idx % RUN_EVERY_N_FRAMES == 0)

        if run_yolo:
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

            new_boxes = []
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

                new_boxes.append((x1, y1, x2, y2, cls, conf))

            # cache detections for use on skipped frames
            last_boxes = new_boxes

        # draw last_boxes on current frame (even if YOLO skipped this frame)
        for x1, y1, x2, y2, cls, conf in last_boxes:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # simple center-of-mass marker for the box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

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
        "<h2>Flyby YOLOv8n Vehicle Stream</h2>"
        "<img src='/video' />"
        "</body></html>"
    )

if __name__ == "__main__":
    print("[INFO] Starting YOLOv8n vehicle stream on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
