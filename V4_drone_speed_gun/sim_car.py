# V4_drone_speed_gun/sim_car.py
import time
import numpy as np
from .geometry import unit_vector_from_heading_deg

class CarSim:
    def __init__(self, road_origin_vec, road_heading_deg, speed_mps):
        self.origin = road_origin_vec
        self.dir_vec = unit_vector_from_heading_deg(road_heading_deg)
        self.speed_mps = speed_mps
        self.t0 = time.time()

    def get_state(self):
        t_now = time.time()
        dt = t_now - self.t0
        s = self.speed_mps * dt
        pos_vec = self.origin + s * self.dir_vec
        return {"t": t_now, "pos_vec": pos_vec}
