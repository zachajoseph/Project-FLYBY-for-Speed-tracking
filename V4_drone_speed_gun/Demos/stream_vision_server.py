#!/usr/bin/env python3
import cv2
from flask import Flask, Response
from V4_drone_speed_gun.vision_model import VisionMeasurementSource

DRONE_ALT_M = 30.0  # pretend drone is 30 m above ground

app = Flask(__name__)
vision = VisionMeasurementSource()

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

frame_idx = 0

def generate_frames():
    global frame_idx
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[STREAM] Camera error or end of stream.")
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[STREAM] frame {frame_idx}, shape={frame.shape}, mean={frame.mean():.2f}")

        meas = vision.estimate_measurement(frame, DRONE_ALT_M)
        if meas is not None:
            R, beta, gamma = meas
            text = f"R={R:5.1f}m, beta={beta:+.2f}rad, gamma={gamma:+.2f}rad"
            color = (0, 255, 0)
        else:
            text = "No car detected"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return (
        "<html><body>"
        "<h2>Flyby Vision Stream</h2>"
        "<img src='/video' />"
        "</body></html>"
    )

if __name__ == "__main__":
    print("[INFO] Starting vision stream on http://0.0.0.0:5000/video")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
