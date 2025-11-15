# Project-FLYBY-for-Speed-tracking

Current V2 set up:

cd drone_speed_gun
pip install -r requirements.txt
python sim_speed_gun.py
- this first listens to the simulated drone over MAVLink
- Simulates a car as a vector along a straight road
- Use the car's position history in vectors
- Prints range between drone and car