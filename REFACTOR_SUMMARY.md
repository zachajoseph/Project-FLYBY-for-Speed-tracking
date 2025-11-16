# V4 Speed Gun Refactor Summary

## Overview
The V4 drone speed gun has been refactored to align with the high-level architecture:
- **Drone pose**: Read from ArduPilot (MAVLink) OR AirSim (no Python-side car simulation)
- **Car simulation**: Exists ONLY in Unreal/AirSim; no Python-side `CarSim` or fake MAVLink connections
- **Vision**: OpenCV detects the car in camera frames (AirSim or real camera)
- **Speed estimation**: Uses vision-detected car position only (no ground truth references)

## Key Changes

### 1. **config.py**
- **Removed**: `CAR_SPEED_MPS`, `RANGE_NOISE_STD_M`, `BETA_NOISE_STD_DEG`, `GAMMA_NOISE_STD_DEG`, `USE_SENSOR_BASED_ESTIMATE`
- **Removed**: QGC-related variables (already commented out)
- **Removed**: `"sitl"` and `"airsim-truth"` mode descriptions
- **Kept**: `ROAD_HEADING_DEG`, `SPEED_LIMIT_MPH`, `OVER_LIMIT_MARGIN_MPH`, `WINDOW_SIZE`, `LOOP_HZ`
- **Clarification**: `MAVLINK_CONNECTION` is now explicitly for drone pose only

### 2. **vision_model.py**
- **Added**: `os` import for environment variable support
- **Removed**: `model_path` and `config_path` constructor parameters
- **Added**: `_load_model(backend: str)` internal method that:
  - Selects between `"model1"` (SSD-Mobilenet v3, default) or `"model2"` (placeholder)
  - Reads backend choice from `VISION_BACKEND` environment variable
  - Applies model-specific initialization
- **Added**: `_detect(frame)` internal method for unified detection interface
- **Modified**: `__init__` now calls `_load_model` based on environment variable
- **Unchanged**: `estimate_measurement(frame, drone_alt_m)` public API signature
- **Note**: `model2` paths are placeholders; update with actual model paths when available

### 3. **speed_gun_v4.py**
- **Removed imports**: `pymavlink.mavutil`, `sensor_from_truth`, `apply_sensor_noise`
- **Added imports**: `format_speed_line` from utils
- **Removed function**: `build_car_source()` (no longer needed)
- **Removed variable usage**: `CAR_SPEED_MPS`, `USE_SENSOR_BASED_ESTIMATE`, `car`, `car_state`, `car_pos_true`, `range_true_m`, `true_mph`, `err_mph`, `last_car_hb`, `car_heading_cdeg`
- **Modified main loop**:
  - Removed all Python-side car simulation references
  - Changed timing from `t_car` (from car state) to `t_now = time.time()`
  - **Vision modes only**: 
    - Get frame from AirSim or real camera
    - Call `vision.estimate_measurement(frame, drone_alt)`
    - If `None`, skip iteration gracefully (no crash)
    - Otherwise, compute `car_pos_est` from measurement only
  - **Non-vision modes**: Print error and exit (car only exists in Unreal)
  - Removed ground-truth range calculations
  - Speed estimation now uses ONLY `car_pos_est` derived from vision
  - Print output simplified to show only estimated values and OVER flag
  - Uses `format_speed_line()` utility for clean formatting

### 4. **utils.py**
- **Modified**: `format_speed_line()` function:
  - **Removed parameters**: `range_true_m`, `true_mph`, `err_mph`
  - **Kept parameters**: `drone_alt`, `drone_heading_deg`, `drone_speed_mps`, `R_meas`, `est_mph`, `over`
  - **Updated output**: Removed references to truth values and errors
  - Cleaner, vision-only output format

### 5. **geometry.py**
- **Unchanged**: `sensor_from_truth()` and `apply_sensor_noise()` remain for potential future use
- **Usage**: These functions are no longer called in the main loop but are kept for backward compatibility

## Verification Checklist

✅ **No QGC integration**
- No QGC IP/port/sysid variables
- No fake car MAVLink connection
- No `GLOBAL_POSITION_INT` for car simulation

✅ **No Python-side CarSim**
- Removed `build_car_source()`, `CarSim` usage
- Removed synthetic sensor path (except in non-vision modes which now error out)
- Only vision-derived `car_pos_est` is used

✅ **Minimal AirSim <-> OpenCV communication**
- `AirsimClient.get_rgb_frame()` → `VisionMeasurementSource.estimate_measurement()` → `(R, beta, gamma)`
- Clean, direct data flow

✅ **Two OpenCV model backends**
- `VISION_BACKEND` environment variable (default: `"model1"`)
- Both backends return `(classes, confidences, boxes)` via `_detect()`
- Both pick best car-like detection and compute `(R, beta, gamma)`
- Placeholder `model2` ready for implementation

✅ **Main loop behavior**
- Drone source: MODE-based (AirSim or MAVLink)
- Vision source: Conditional on `"vision"` in MODE
- Handles `None` returns gracefully
- Computes speed from vision measurements only
- Prints simplified status line

✅ **Stable interfaces**
- `MavlinkClient.get_drone_position_vec()` — unchanged
- `AirsimClient.get_drone_position_vec()` — unchanged
- `AirsimClient.get_rgb_frame()` — unchanged
- `rel_vec_from_sensor(R, beta, gamma)` — unchanged
- `PositionSpeedEstimator.update(t, car_pos_vec)` — unchanged
- `VisionMeasurementSource.estimate_measurement(frame, drone_alt_m)` — unchanged

## Running the Refactored Code

```bash
# With default backend (model1)
python -m V4_drone_speed_gun.speed_gun_v4

# With alternative backend
VISION_BACKEND=model2 python -m V4_drone_speed_gun.speed_gun_v4
```

Expected output in `airsim-vision` mode:
```
[AIRSIM] Connected as Drone1
[VISION] Using backend: model1
[SIM] Mode = airsim-vision
[SIM] Starting loop. Ctrl+C to quit.
[SpeedGun] Drone (alt  50.0 m, heading 123.5°,  8.5 m/s)   Range ~285.3 m   Car est  42.3 mph   OVER=True
[SpeedGun] Drone (alt  50.1 m, heading 124.2°,  8.6 m/s)   Range ~287.1 m   Car est  42.1 mph   OVER=True
...
```

## Notes

- The refactored code assumes AirSim spawns and moves the car as an actor/mesh
- Vision detection is the only source of car position information
- Speed limits and margins are still configurable in `config.py`
- Drone velocity/heading calculations are independent of car state
- All timestamps now use `time.time()` for consistency
