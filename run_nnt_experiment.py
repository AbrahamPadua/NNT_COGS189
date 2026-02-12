import os
import sys
import time
import json
import glob
import random
from queue import Queue
from threading import Thread, Event

import numpy as np
from psychopy import visual, core
from psychopy.hardware import keyboard
from psychopy.visual.movie import MovieStim
from serial import Serial
import serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams

# -----------------------------
# Experiment configuration
# -----------------------------
CYTON_BOARD_ID = 0  # 0 if no daisy, 2 if daisy, 6 if daisy+wifi
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'

SCREEN_WIDTH = 1536
SCREEN_HEIGHT = 864
FULLSCREEN = True

PRE_STIM_DURATION = 0.7
STIM_DURATION = 5.0
TRIALS_PER_SCENARIO = 100
SEED = 1

SUBJECT = 1
SESSION = 1
RUN = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
DATA_DIR = os.path.join(BASE_DIR, "data", f"sub-{SUBJECT:02d}", f"ses-{SESSION:02d}", f"run-{RUN:02d}")

CATEGORIES = ["Human", "AI", "Robot"]
SCENARIOS = ["Neutral", "Happy", "Sad", "Pain"]
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

# -----------------------------
# Cyton / Dongle helpers
# -----------------------------

def find_openbci_port():
    """Find the port to which the Cyton dongle is connected."""
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Error finding ports on your operating system')

    openbci_port = ''
    for port in ports:
        try:
            s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            line = ''
            time.sleep(2)
            if s.in_waiting:
                c = ''
                while '$$$' not in line:
                    c = s.read().decode('utf-8', errors='replace')
                    line += c
                if 'OpenBCI' in line:
                    openbci_port = port
            s.close()
        except (OSError, serial.SerialException):
            pass

    if openbci_port == '':
        raise OSError('Cannot find OpenBCI port.')
    return openbci_port


def start_cyton_stream():
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
    else:
        params.ip_port = 9000

    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.config_board('/0')
    board.config_board('//')
    board.config_board(ANALOGUE_MODE)
    board.start_stream(45000)
    return board


def start_data_thread(board, stop_event, queue_in):
    eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID)
    aux_channels = board.get_analog_channels(CYTON_BOARD_ID)
    ts_channel = board.get_timestamp_channel(CYTON_BOARD_ID)

    def get_data():
        while not stop_event.is_set():
            data_in = board.get_board_data()
            timestamp_in = data_in[ts_channel]
            eeg_in = data_in[eeg_channels]
            aux_in = data_in[aux_channels]
            if len(timestamp_in) > 0:
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.05)

    thread = Thread(target=get_data, daemon=True)
    thread.start()
    return thread


# -----------------------------
# Video helpers
# -----------------------------

def collect_video_files():
    scenario_defs = []
    video_files = {}
    scenario_id = 0

    for category in CATEGORIES:
        for scenario in SCENARIOS:
            folder = os.path.join(VIDEO_DIR, category, scenario)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")

            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(VIDEO_EXTS)
            ]

            scenario_defs.append({
                "id": scenario_id,
                "category": category,
                "scenario": scenario,
                "folder": folder,
            })
            video_files[scenario_id] = files
            scenario_id += 1

    missing = [s for s in scenario_defs if len(video_files[s["id"]]) == 0]
    if missing:
        missing_names = ", ".join([f"{m['category']}/{m['scenario']}" for m in missing])
        raise FileNotFoundError(
            f"No videos found for: {missing_names}.\n"
            f"Add videos (mp4/mov/avi/mkv) to each scenario folder in {VIDEO_DIR}."
        )

    return scenario_defs, video_files


def build_trial_sequence(num_scenarios):
    trial_sequence = []
    for scenario_id in range(num_scenarios):
        trial_sequence.extend([scenario_id] * TRIALS_PER_SCENARIO)
    random.seed(SEED)
    random.shuffle(trial_sequence)
    return trial_sequence


# -----------------------------
# Main experiment
# -----------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    scenario_defs, video_files = collect_video_files()
    trial_sequence = build_trial_sequence(len(scenario_defs))

    kb = keyboard.Keyboard()
    window = visual.Window(
        size=[SCREEN_WIDTH, SCREEN_HEIGHT],
        fullscr=FULLSCREEN,
        allowGUI=False,
        checkTiming=True,
    )

    fixation = visual.TextStim(window, text='+', height=0.15, color='white')

    stop_event = Event()
    queue_in = Queue()

    board = start_cyton_stream()
    data_thread = start_data_thread(board, stop_event, queue_in)

    eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID)
    aux_channels = board.get_analog_channels(CYTON_BOARD_ID)

    eeg = np.zeros((len(eeg_channels), 0))
    aux = np.zeros((len(aux_channels), 0))
    timestamp = np.zeros((0,))
    markers = []

    def drain_queue():
        nonlocal eeg, aux, timestamp
        while not queue_in.empty():
            eeg_in, aux_in, ts_in = queue_in.get()
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            aux = np.concatenate((aux, aux_in), axis=1)
            timestamp = np.concatenate((timestamp, ts_in), axis=0)

    try:
        total_trials = len(trial_sequence)
        for i_trial, scenario_id in enumerate(trial_sequence, start=1):
            scenario = scenario_defs[scenario_id]
            video_path = random.choice(video_files[scenario_id])

            fixation.draw()
            window.flip()
            core.wait(PRE_STIM_DURATION)

            drain_queue()
            marker = {
                "trial_index": i_trial,
                "scenario_id": scenario_id,
                "category": scenario["category"],
                "scenario": scenario["scenario"],
                "video": os.path.relpath(video_path, BASE_DIR),
                "stim_start_time": time.time(),
                "start_sample_index": int(eeg.shape[1]),
            }

            movie = MovieStim(window, video_path, loop=False, noAudio=True)
            movie.play()
            stim_clock = core.Clock()
            while stim_clock.getTime() < STIM_DURATION:
                if 'escape' in kb.getKeys():
                    raise KeyboardInterrupt
                movie.draw()
                window.flip()

            movie.stop()
            drain_queue()
            marker["stim_end_time"] = time.time()
            marker["end_sample_index"] = int(eeg.shape[1])
            markers.append(marker)

            print(f"Trial {i_trial}/{total_trials} - {scenario['category']} {scenario['scenario']}")

    except KeyboardInterrupt:
        print("Experiment interrupted by user.")

    finally:
        stop_event.set()
        board.stop_stream()
        board.release_session()
        window.close()

        drain_queue()
        np.save(os.path.join(DATA_DIR, "eeg.npy"), eeg)
        np.save(os.path.join(DATA_DIR, "aux.npy"), aux)
        np.save(os.path.join(DATA_DIR, "timestamp.npy"), timestamp)
        with open(os.path.join(DATA_DIR, "markers.json"), "w", encoding="utf-8") as f:
            json.dump(markers, f, indent=2)
        with open(os.path.join(DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "subject": SUBJECT,
                "session": SESSION,
                "run": RUN,
                "pre_stim_duration": PRE_STIM_DURATION,
                "stim_duration": STIM_DURATION,
                "trials_per_scenario": TRIALS_PER_SCENARIO,
                "seed": SEED,
                "categories": CATEGORIES,
                "scenarios": SCENARIOS,
            }, f, indent=2)

        print(f"Saved data to {DATA_DIR}")


if __name__ == "__main__":
    main()
