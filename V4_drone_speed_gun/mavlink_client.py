# V4_drone_speed_gun/mavlink_client.py
import numpy as np
from pymavlink import mavutil

class MavlinkClient:
    def __init__(self, connection_str):
        print(f"[MAVLINK] Connecting to {connection_str} ...")
        self.master = mavutil.mavlink_connection(connection_str)
        self.master.wait_heartbeat()
        print("[MAVLINK] Heartbeat received.",
              "System:", self.master.target_system,
              "Component:", self.master.target_component)

        self.last_pos_vec = None
        self.origin_lat = None
        self.origin_lon = None

    def get_drone_position_vec(self):
        msg = self.master.recv_match(type="LOCAL_POSITION_NED",
                                     blocking=True, timeout=1.0)
        if msg is None:
            return self.last_pos_vec

        self.last_pos_vec = np.array(
            [float(msg.x), float(msg.y), -float(msg.z)], dtype=float
        )
        return self.last_pos_vec

    def get_origin_latlon(self):
        if self.origin_lat is not None:
            return self.origin_lat, self.origin_lon

        print("[MAVLINK] Waiting for GLOBAL_POSITION_INT ...")
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT",
                                     blocking=True, timeout=10.0)
        if msg is None:
            raise RuntimeError("No GLOBAL_POSITION_INT received")

        self.origin_lat = msg.lat * 1e-7
        self.origin_lon = msg.lon * 1e-7
        print(f"[MAVLINK] Origin lat/lon: "
              f"{self.origin_lat:.7f}, {self.origin_lon:.7f}")
        return self.origin_lat, self.origin_lon
