#!/usr/bin/env python3
import cv2
import time
import numpy as np
import os

from V4_drone_speed_gun.vision_model import VisionMeasurementSource

DRONE_ALT_M = 30.0  # pretend drone is 30 m above ground

def main(source=0):
    # If you want to force V4L2 instead of GStreamer, you can try:
    # cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERR] Could not open video source: {source}")
        return

    vision = VisionMeasurementSource()

    # Headless detection: no DISPLAY -> no GUI
    headless = os.environ.get("DISPLAY", "") == ""
    if headless:
        print("[INFO] Running in headless mode (no cv2.imshow).")
    else:
        print("[INFO] GUI mode: Press ESC to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[TEST] End of stream or camera error.")
            break

        meas = vision.estimate_measurement(frame, DRONE_ALT_M)

        if meas is not None:
            R, beta, gamma = meas
            msg = f"R={R:5.1f}m, beta={beta:+.2f}rad, gamma={gamma:+.2f}rad"
        else:
            msg = "No car detected"

        # Always log to console
        if frame_count % 10 == 0:
            print(f"[VISION] {msg}")
        frame_count += 1

        # Only try imshow if we actually have a display
        if not headless:
            cv2.putText(frame, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Vision smoke test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(source=0)
