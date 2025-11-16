#!/usr/bin/env python3
import os
import sys
# Ensure project root is on sys.path so imports like `V4_drone_speed_gun` work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import time
import numpy as np

from V4_drone_speed_gun.vision_model import VisionMeasurementSource

# Pick a fake altitude for testing geometry
DRONE_ALT_M = 30.0  # pretend drone is 30 m above ground

def main(source=1):
    # source can be 0 (webcam) or a video path string
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERR] Could not open video source: {source}")
        return

    vision = VisionMeasurementSource()

    print("[TEST] Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[TEST] End of stream or camera error.")
            break

        # Run your existing measurement function
        meas = vision.estimate_measurement(frame, DRONE_ALT_M)

        # Simple visual debug: draw a text overlay if we got a measurement
        if meas is not None:
            R, beta, gamma = meas
            text = f"R={R:5.1f}m, beta={beta:+.2f}rad, gamma={gamma:+.2f}rad"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No car detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Vision smoke test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change this to a filename like "test_cars.mp4" to test a recorded video
    main(source=0)
