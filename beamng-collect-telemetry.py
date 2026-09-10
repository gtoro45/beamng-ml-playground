from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Electrics
import beamngpy.connection.connection as bng_conn

import pygetwindow as gw
import mss
from PIL import Image
import itertools

import queue
import threading
import csv
import argparse

# Force the socket handshake string to match the game engine's expected version
bng_conn.Connection.PROTOCOL_VERSION = 'v1.26'

################################## PARSE CLI ARGS ##################################


################################## SETUP CODE ##################################
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
WRITE_QUEUE = queue.Queue

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
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    
    # write to queue
    WRITE_QUEUE.put(item=(steering, throttle, brake, speed, img, frame_no))

# read from thread-safe queue to handle IO tasks separately
def writer_thread():
    while True:
        # get and validate entry
        entry = WRITE_QUEUE.get()
        if entry is None: 
            break
        
        # downscale image based on CLI args
        # TODO
        
        # write to disk
        
################################## CAPTURE LOOP ##################################
SIM_TICKS = 6   # define the tickrate

sct = mss.MSS()                 # capture library for computer vision
bng_win = get_beamng_window()   # beamng window
bbox = {                        # the bounding box for the current BeamNG window (assumes no movement)
    "top": bng_win.top,
    "left": bng_win.left,
    "width": bng_win.width,
    "height": bng_win.height,
}

# spin up the writer thread
writer = threading.Thread(target=writer_thread)
writer.start()

# start the collection loop
print("Hit enter when done...")
try:
    for frame in itertools.count():
        # advance simulation by SIM_TICKS steps
        bng.control.step(SIM_TICKS)
        
        # capture data
        capture_data(sct, bbox, vehicle, frame)
        
except KeyboardInterrupt:
    print("Ending Capture")
finally:
    # join the writer thread
    WRITE_QUEUE.put(None)   # None kills writer thread loop
    writer.join()
    bng.close()

#####################################################################################