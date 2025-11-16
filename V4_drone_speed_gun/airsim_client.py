# V4_drone_speed_gun/airsim_client.py
import numpy as np
import airsim

class AirsimClient:
    def __init__(self, vehicle_name="Drone1"):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        self.vehicle = vehicle_name
        print("[AIRSIM] Connected as", vehicle_name)

    def get_drone_position_vec(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle)
        pos = state.kinematics_estimated.position
        # AirSim uses NED: x=north, y=east, z=down
        return np.array([pos.x_val, pos.y_val, -pos.z_val], dtype=float)

    def get_rgb_frame(self, camera_name="0"):
        resp = self.client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene,
                                 pixels_as_float=False, compress=False)],
            vehicle_name=self.vehicle,
        )[0]
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img_bgr = img1d.reshape(resp.height, resp.width, 3)
        return img_bgr

    # Optional: if you spawn an AirSim car and want its ground truth
    def get_car_position_vec(self, car_name="Car1"):
        pose = self.client.simGetObjectPose(car_name)
        pos = pose.position
        return np.array([pos.x_val, pos.y_val, -pos.z_val], dtype=float)
