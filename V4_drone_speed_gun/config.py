# V4_drone_speed_gun/config.py

# --- Backends / modes -------------------------------------------------
# "airsim-vision":  AirSim drone + camera vision detection
# "camera-vision":  Real camera + MAVLink drone, OpenCV detection
MODE = "airsim-vision"

# For non-AirSim camera (e.g. USB cam)
VIDEO_SOURCE = 0     # cv2.VideoCapture index or RTSP URL

# --- MAVLink (for drone pose only) ------------------------------------
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

# --- Road / car config ------------------------------------------------
ROAD_HEADING_DEG = 0.0
SPEED_LIMIT_MPH = 35.0
OVER_LIMIT_MARGIN_MPH = 5.0

WINDOW_SIZE = 15
LOOP_HZ = 10.0
