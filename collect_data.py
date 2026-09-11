from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Electrics
import beamngpy.connection.connection as bng_conn

import pygetwindow as gw
import mss
from PIL import Image
import itertools

import queue
import threading
import argparse
import numpy as np
import pickle

from load_data import view_data

# Force the socket handshake string to match the game engine's expected version
bng_conn.Connection.PROTOCOL_VERSION = 'v1.26'

################################## PARSE CLI ARGS ##################################
parser = argparse.ArgumentParser()
parser.add_argument('--dims', nargs=2, type=int, help="Dimensions of the captured image for model input (tuple)", required=True)
parser.add_argument('--model', type=str, help="Metadata: name of the model for which this dataset will be used for", required=True)
parser.add_argument('--show', action="store_true", help="Preview the dataset in a scrollable matplotlib window")
parser.add_argument('--verbose', action="store_true", help="Verbose capture output in terminal")
args = parser.parse_args()
DIMS = args.dims
MODEL = args.model
POST_PROC_VIEW = args.show
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
scenario = Scenario('west_coast_usa', 'example')

# spawn a vehicle
vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')

# attach Electronics sensors
electrics = Electrics()
vehicle.sensors.attach('electrics', electrics)

# add vehicle to scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))

# place files defining our scenario for the simulator to read
scenario.make(bng)

# set to 60fps lock, and pause the game for iterative capture
bng.settings.set_deterministic(60)
bng.control.pause()

# Load and start scenario
bng.scenario.load(scenario)
bng.scenario.start()

# Make the vehicle's AI span the map
vehicle.ai.set_mode('traffic')


################################## CAPTURE FUNCTIONS ###################################
WRITE_QUEUE = queue.Queue()
DATA_OUTPUT_PATH = f"beamng_dataset-{MODEL}.pkl"

# find the window coordinates of BeamNG Drive
def get_beamng_window():
    windows = gw.getWindowsWithTitle("BeamNG")
    if not windows:
        raise RuntimeError("BeamNG window not found. Ensure the game is running.")
    return windows[0]

# data capturing function, once per sim step
def capture_data(sct: mss.MSS, bbox: dict, vehicle: Vehicle, frame_no: int):
    # capture vehicle data
    vehicle.sensors.poll()
    steering = electrics['steering_input']
    throttle = electrics['throttle_input']
    brake = electrics['brake_input']
    speed = electrics['wheelspeed']    
    
    # capture image
    sct_img = sct.grab(bbox)
    
    # write to queue
    WRITE_QUEUE.put(item=(steering, throttle, brake, speed, sct_img, frame_no))

# read from thread-safe queue to handle IO tasks separately
def writer_thread():
    data_list = []
    labels_list = []
    
    # consumer loop
    while True:
        # get and validate entry
        entry = WRITE_QUEUE.get()
        if entry is None: break
        steering, throttle, brake, speed, sct_img, frame_no = entry
        
        # Convert and downscale image based on CLI args
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        res = img.resize(DIMS)
        
        # Match CIFAR dataset structure (H, W, 3) --> (3, H, W) --> flatten to 1D
        img_arr = np.array(res, dtype=np.uint8)         # (H, W, 3) (original)
        img_planar = np.transpose(img_arr, (2, 0, 1))   # (3, H, W) (cifar)
        flat_img = img_planar.flatten()                 # flatten to shape 3 * H * W
        
        # multi-dimensional telemetry labels
        label_vec = np.array([steering, throttle, brake, speed], dtype=np.float32)
        
        # write to parallel data and label lists
        data_list.append(flat_img)
        labels_list.append(label_vec)
        WRITE_QUEUE.task_done()     # thread safety
        if VERBOSE: print(f"[VERBOSE] Logged {(steering, throttle, brake, speed, sct_img, frame_no)}")
    
    # consolidate and write dataset to disk
    if data_list:
        dataset = {
            b'data': np.vstack(data_list),       # shape: (N, 3 * H * W)
            b'labels': np.vstack(labels_list),   # shape: (N, 4)
            b'dims': DIMS                        # metadata: (width, height)
        }
        
        output_path = DATA_OUTPUT_PATH
        with open(output_path, 'wb') as file:
            pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {len(data_list)} frames to {output_path}")
        
################################## CAPTURE LOOP ##################################
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

# spin up the writer thread
writer = threading.Thread(target=writer_thread)
writer.start()

# start the collection loop
print("Running capture...")
try:
    for frame in itertools.count():
        bng.control.step(SIM_TICKS)             # step simuilation forward
        if frame < 30: continue                 # allow rendering to warm up before collection (30 frames = 3 seconds @ 60FPS with SIM_TICKS = 6)
        capture_data(sct, bbox, vehicle, frame) # capture data
        
except KeyboardInterrupt:
    print("Ending Capture")
finally:
    # join the writer thread
    WRITE_QUEUE.put(None)   # None kills writer thread loop
    writer.join()
    bng.close()
    if POST_PROC_VIEW: view_data(DATA_OUTPUT_PATH, DIMS)

#####################################################################################
