# estimators.py

from collections import deque
import numpy as np
import utils


class PositionSpeedEstimator:
    """
    Uses a vector of car positions over time to estimate speed along the road.
    - Projects car position onto road direction.
    - Uses finite difference on the last two samples.
    """
    def __init__(self, road_heading_deg, window_size=15):
        self.dir_vec = utils.unit_vector_from_heading_deg(road_heading_deg)
        self.samples = deque(maxlen=window_size)  # (t, s_along_road_m)

    def update(self, t, car_pos_vec):
        """
        Add a new (t, car_pos_vec) sample and return estimated
        road speed in m/s, or None if not enough samples yet.
        """
        car_pos_vec = np.array(car_pos_vec, dtype=float)

        # Project car position onto road direction (scalar)
        s = float(np.dot(car_pos_vec, self.dir_vec))
        self.samples.append((t, s))

        if len(self.samples) < 2:
            return None

        (t1, s1), (t2, s2) = self.samples[-2], self.samples[-1]
        dt = t2 - t1
        if dt <= 0:
            return None

        v_road_mps = (s2 - s1) / dt
        return v_road_mps
