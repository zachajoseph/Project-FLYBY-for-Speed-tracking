# config.py

# MAVLink connection from Jetson to SITL/MAVProxy
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

# QGC laptop IP (where QGroundControl is running)
QGC_IP = "192.168.1.220"   # change if your laptop IP changes
QGC_PORT = 14550
CAR_SYSID = 42             # arbitrary system id for the car

# Road and car configuration
ROAD_HEADING_DEG = 0.0        # 0° = +X (north in NED)
CAR_SPEED_MPS = 20.0          # ~44.7 mph
SPEED_LIMIT_MPH = 35.0
OVER_LIMIT_MARGIN_MPH = 5.0   # how much over before we flag

# Smoothing / estimator settings
WINDOW_SIZE = 15              # history length
LOOP_HZ = 10.0                # main loop rate

# Sensor model (synthetic range + angles derived from truth)
RANGE_NOISE_STD_M = 0.5       # m
BETA_NOISE_STD_DEG = 0.5      # deg (elevation from nadir)
GAMMA_NOISE_STD_DEG = 1.0     # deg (azimuth)

# True = use reconstructed car_pos_est, False = use car truth
USE_SENSOR_BASED_ESTIMATE = True
