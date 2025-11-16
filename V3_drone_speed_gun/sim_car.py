# sim_car.py

import time
import numpy as np
from . import utils  # if you make this a package
# or just: import utils   if you run from this folder without a package

# assuming simple `import utils` style:
import utils


class CarSim:
    """
    Very simple car simulator:
    - Car moves along a straight road at constant speed.
    - Road defined by origin vector + heading.
    """
    def __init__(self, road_origin_vec, road_heading_deg, speed_mps):
        self.origin = road_origin_vec
        self.dir_vec = utils.unit_vector_from_heading_deg(road_heading_deg)
        self.speed_mps = speed_mps
        self.t0 = time.time()

    def get_state(self):
        """
        Returns:
        {
            "t": timestamp (s),
            "pos_vec": np.array([x,y,z]) in local frame
        }
        """
        t_now = time.time()
        dt = t_now - self.t0
        s = self.speed_mps * dt   # distance travelled along road
        pos_vec = self.origin + s * self.dir_vec
        return {"t": t_now, "pos_vec": pos_vec}
