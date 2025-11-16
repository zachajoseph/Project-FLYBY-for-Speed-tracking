#!/usr/bin/env python3
import cv2
import time
import os
from flask import Flask, Response

from V4_drone_speed_gun.vision_model import VisionMeasurementSource

DRONE_ALT_M = 30.0  # pretend drone is 30 m above ground

app = Flask(__name__)

# Change this if your camera index is different
CAM_SOURCE = 0 

cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"[ERR] Could not open video source: {CAM_SOURCE}")
    cap = None

vision = VisionMeasurementSource()

frame_idx = 0

def generate_frames():
    global frame_idx
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            print("[STREAM] Camera error or end of stream.")
            break

        if frame_idx % 30 == 0:
            print(f"[STREAM] frame {frame_idx}, shape={frame.shape}, mean={frame.mean():.2f}")


        # Run detection / measurement
        meas = vision.estimate_measurement(frame, DRONE_ALT_M)
        if meas is not None:
            R, beta, gamma = meas
            text = f"R={R:5.1f}m, beta={beta:+.2f}rad, gamma={gamma:+.2f}rad"
            color = (0, 255, 0)
        else:
            text = "No car detected"
            color = (0, 0, 255)

        # Overlay text for visualization
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG stream chunk
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    # Bind to 0.0.0.0 so other machines can reach it
    print("[INFO] Starting vision stream on http://0.0.0.0:5000/video")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
