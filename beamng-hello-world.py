from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import beamngpy.connection.connection as bng_conn

import pygetwindow as gw
import mss
from PIL import Image

# Force the socket handshake string to match the game engine's expected version
bng_conn.Connection.PROTOCOL_VERSION = 'v1.26'

# globals
BEAMNG_DRIVE_PATH=r"H:\Games Library\Steam Library\steamapps\common\BeamNG.drive"

# launch the beamng process
bng = BeamNGpy(
    host="localhost", 
    port=25252,
    home=BEAMNG_DRIVE_PATH
)
bng.open()

# create a scenario
scenario = Scenario('west_coast_usa', 'example')

# spawn a vehicle
vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')

# add vehicle to scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))

# place files defining our scenario for the simulator to read
scenario.make(bng)
bng.settings.set_deterministic(60)
bng.control.pause()

# Load and start scenario
bng.scenario.load(scenario)
bng.scenario.start()

# Make the vehicle's AI span the map
vehicle.ai.set_mode('traffic')

# collect camera data (via Python, beamngpy doesn't support Camera module for .drive, only .tech)
def get_beamng_window():
    windows = gw.getWindowsWithTitle("BeamNG")
    if not windows:
        raise RuntimeError("BeamNG window not found. Ensure the game is running.")
    return windows[0]

sct = mss.MSS()                 # capture library for computer vision
bng_win = get_beamng_window()   # beamng window

for i in range(41):
    bng.control.step(10)

    if i % 10 == 0:
        # Define the bounding box from the live window coordinates
        bbox = {
            "top": bng_win.top,
            "left": bng_win.left,
            "width": bng_win.width,
            "height": bng_win.height,
        }

        # Grab only that window's region
        sct_img = sct.grab(bbox)
        
        # Convert raw BGRA frame to a standard PIL RGB Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.save(f"img-{i}.png")

# kill program on terminal input
input('Hit Enter when done...')
bng.close()