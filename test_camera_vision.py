#!/usr/bin/env python3
import cv2
import time

from V4_drone_speed_gun.vision_model import VisionMeasurementSource

# Fake altitude (meters) for the math; doesn't affect detection itself
DRONE_ALT_M = 30.0

def main(source=0):
    # source=0 => default laptop camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERR] Could not open video source: {source}")
        return

    vision = VisionMeasurementSource()

    print("[TEST] Running live camera vision test.")
    print("[TEST] Press ESC in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[TEST] No frame from camera, exiting.")
            break

        # Run your existing measurement function (frame, altitude)
        meas = vision.estimate_measurement(frame, DRONE_ALT_M)

        if meas is not None:
            R, beta, gamma = meas
            status_text = f"R={R:5.1f}m  beta={beta:+.2f}rad  gamma={gamma:+.2f}rad"
            color = (0, 255, 0)
        else:
            status_text = "No car-like detection"
            color = (0, 0, 255)

        # Draw text on the frame so you see what's happening
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Camera Vision Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 0 = default laptop camera
    main(source=0)
