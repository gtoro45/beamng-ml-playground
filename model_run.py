from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics
import beamngpy.connection.connection as bng_conn

from PIL import Image
import pygetwindow as gw
import mss
import itertools

import argparse
import numpy as np
import pickle


from load_data import view_data

# Force the socket handshake string to match the game engine's expected version
bng_conn.Connection.PROTOCOL_VERSION = 'v1.26'

################################## PARSE CLI ARGS ##################################
parser = argparse.ArgumentParser()
parser.add_argument('--dims', nargs=2, type=int, help="Dimensions of the captured image for model input (tuple)", required=True)
parser.add_argument('--verbose', action="store_true", help="Verbose capture output in terminal")
args = parser.parse_args()
DIMS = args.dims
VERBOSE = args.verbose

################################## SETUP CODE ##################################
BEAMNG_DRIVE_PATH=r"H:\Games Library\Steam Library\steamapps\common\BeamNG.drive"

# launch the beamng process
bng = BeamNGpy(
    host="localhost", 
    port=25252,
    home=BEAMNG_DRIVE_PATH
)
bng.open()
bng.hide_hud()

# create a scenario
# scenario = Scenario('west_coast_usa', 'example')
scenario = Scenario('east_coast_usa', 'example')

# spawn a vehicle
vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')

# attach Electronics sensors
electrics = Electrics()
vehicle.sensors.attach('electrics', electrics)

# add vehicle to scenario at this position and rotation
# scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))    # west coast
scenario.add_vehicle(vehicle, pos=(-426.68, -43.59, 31.11), rot_quat=(0, 0, 1, 0))

# place files defining our scenario for the simulator to read
scenario.make(bng)

# set to 60fps lock, and pause the game for iterative capture
bng.settings.set_deterministic(60)
bng.control.pause()

# Load and start scenario
bng.scenario.load(scenario)
bng.scenario.start()

################################## HELPER FUNCTIONS ###################################
# find the window coordinates of BeamNG Drive
def get_beamng_window():
    windows = gw.getWindowsWithTitle("BeamNG")
    if not windows:
        raise RuntimeError("BeamNG window not found. Ensure the game is running.")
    return windows[0]

################################## SIMULATION LOOP ##################################
# Sim runs deterministically at 60 physics steps (ticks) per in-game second.
# Stepping SIM_TICKS = 6 advances in-game time by 6 / 60 = 0.10 seconds per loop.
# Capturing once per step yields 60 / SIM_TICKS = 10 captures per in-game second (10 Hz).
SIM_TICKS = 6

sct = mss.MSS()                 # capture library for computer vision
bng_win = get_beamng_window()   # beamng window

# Define the square size (e.g., 650x650 pixels works well for a 1080p window)
crop_dim = int(bng_win.height * 0.60)  # ~648 pixels
bbox = {
    # Starts above the car to capture the road ahead down through the lane markers
    "top": int(bng_win.top + bng_win.height * 0.28),
    # Symmetrically centered on the car (center line is bng_win.left + bng_win.width * 0.5)
    "left": int(bng_win.left + (bng_win.width / 2) - (crop_dim / 2)),
    "width": crop_dim,
    "height": crop_dim,
}

# load the model and start the collection loop
from models.LeNetCifar10_Speed import device, LeNet_Cifar10_Speed
import torch
import torch.nn as nn

model = LeNet_Cifar10_Speed(
    criterion=nn.SmoothL1Loss(),
    batch_norm=True,
    dropout=True
)
checkpoint = torch.load('lenet-speed.pt', map_location='cpu')
model.model.load_state_dict(checkpoint, strict=True)
model.model.eval()

print("Running simulation...")
try:
    for frame in itertools.count():
        # step simuilation forward
        bng.control.step(SIM_TICKS)             
        
        # Give the car a nudge for the first 5 seconds (50 frames @ 0.10s per step)
        if frame < 50:
            vehicle.control(throttle=0.3, steering=0.0, brake=0.0)
            continue
        
        # (1) Prepare model inputs from simulation frame
        # grab image
        sct_img = sct.grab(bbox)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        res = img.resize(DIMS)
        img_arr = np.array(res, dtype=np.uint8)         # (H, W, 3) (original)
        img_planar = np.transpose(img_arr, (2, 0, 1))   # (3, H, W) (cifar)
        flat_img = img_planar.flatten()                 # flatten to shape 3 * H * W
        
        # grab speed
        vehicle.sensors.poll()
        speed = electrics['wheelspeed']
        
        # add batch dimension --> becomes (1, 3, H, W)
        # Note: training was with (N, 3, H, W), where N is the number of samples
        #       with batch_norm=True, this was a batch size of 128
        #       with batch_norm=True, this was a batch size of len(data set)
        # For model input, we just have a single frame to process at a time, hence 1
        input_img_tensor = torch.from_numpy(img_planar).float().unsqueeze(0).to(device) 
        input_speed_tensor = torch.tensor([[speed]], dtype=torch.float32, device=device)
        
        
        # (2) Run model prediction
        with torch.no_grad(): output = model.model(input_img_tensor, input_speed_tensor)
        steering = output[0, 0].item()  # (0, 0) corresponds to batch index 0, output index 0 (steering)
        throttle = output[0, 1].item()  # (0, 1) corresponds to batch index 0, output index 1 (throttle)
        brake = output[0, 2].item()     # (0, 2) corresponds to batch index 0, output index 2 (brake)
        
        # (3) Apply predicted inputs to vehicle
        vehicle.control(steering=steering, throttle=throttle, brake=brake)
        
        # Optional logging
        if VERBOSE: print(f"[VERBOSE] Frame {frame} | Steering: {steering:.2f}, Throttle: {throttle:.2f}, Brake: {brake:.2f}")
        
except KeyboardInterrupt:
    print("Ending Capture")
finally:
    bng.close()

#####################################################################################
