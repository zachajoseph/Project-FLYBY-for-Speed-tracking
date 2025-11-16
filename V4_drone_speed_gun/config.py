# V4_drone_speed_gun/config.py

# --- Backends / modes -------------------------------------------------
# "sitl":      MAVLink + synthetic sensor + CarSim (what you have now)
# "airsim-truth": AirSim for drone + car truth, synthetic sensor
# "airsim-vision": AirSim for drone + camera, OpenCV for car measurement
# "camera-vision": Real camera + MAVLink drone, OpenCV for car measurement
MODE = "sitl"

# For non-AirSim camera (e.g. USB cam)
VIDEO_SOURCE = 0     # cv2.VideoCapture index or RTSP URL

# --- MAVLink / QGC ----------------------------------------------------
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"
QGC_IP  = "192.168.1.220"
QGC_PORT = 14550
CAR_SYSID = 42

# --- Road / car config ------------------------------------------------
ROAD_HEADING_DEG = 0.0
CAR_SPEED_MPS = 20.0
SPEED_LIMIT_MPH = 35.0
OVER_LIMIT_MARGIN_MPH = 5.0

WINDOW_SIZE = 15
LOOP_HZ = 10.0

# --- Sensor model -----------------------------------------------------
RANGE_NOISE_STD_M      = 0.5
BETA_NOISE_STD_DEG     = 0.5
GAMMA_NOISE_STD_DEG    = 1.0

USE_SENSOR_BASED_ESTIMATE = True
