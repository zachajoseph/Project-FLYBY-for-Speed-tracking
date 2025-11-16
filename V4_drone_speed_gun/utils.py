# V4_drone_speed_gun/utils.py
import time

def log(msg: str):
    """
    Tiny logging helper; swap out for 'rich' or 'logging' later if you want.
    """
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


class LoopTimer:
    """
    Helper to run a loop at approximately fixed frequency.

    Example:
        loop = LoopTimer(10.0)  # 10 Hz
        while True:
            loop.start()
            # ... do work ...
            loop.sleep_to_rate()
    """
    def __init__(self, hz: float):
        self.period = 1.0 / float(hz)
        self._t_start = None

    def start(self):
        self._t_start = time.time()

    def sleep_to_rate(self):
        if self._t_start is None:
            return
        elapsed = time.time() - self._t_start
        remaining = self.period - elapsed
        if remaining > 0:
            time.sleep(remaining)


def format_speed_line(
    drone_alt: float,
    drone_heading_deg,
    drone_speed_mps: float,
    R_meas: float,
    est_mph: float,
    over: bool,
) -> str:
    """
    Build the human-readable status line for the console.
    Keeps formatting in one place so main loop stays clean.
    """
    heading_str = "n/a"
    if drone_heading_deg is not None:
        heading_str = f"{drone_heading_deg:5.1f}°"

    return (
        f"[SpeedGun] "
        f"Drone (alt {drone_alt:5.1f} m, heading {heading_str}, {drone_speed_mps:4.1f} m/s)   "
        f"Range ~{R_meas:6.1f} m   "
        f"Car est {est_mph:5.1f} mph   "
        f"OVER={over}"
    )
