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
DEMO_MODE = True  # Set True to run without connecting to Cyton

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

    for category in CATEGORIES:
        for scenario in SCENARIOS:
            folder = os.path.join(VIDEO_DIR, category, scenario)
            if not os.path.isdir(folder):
                print(f"Skipping scenario {category}/{scenario}: missing folder ({folder})")
                continue

            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(VIDEO_EXTS)
            ]

            accessible_files = []
            for file_path in files:
                try:
                    with open(file_path, "rb"):
                        pass
                    accessible_files.append(file_path)
                except OSError:
                    print(f"Ignoring inaccessible video: {file_path}")

            if len(accessible_files) == 0:
                print(f"Skipping scenario {category}/{scenario}: no accessible videos")
                continue

            scenario_id = len(scenario_defs)

            scenario_defs.append({
                "id": scenario_id,
                "category": category,
                "scenario": scenario,
                "folder": folder,
            })
            video_files[scenario_id] = accessible_files

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
    demo_mode = DEMO_MODE
    os.makedirs(DATA_DIR, exist_ok=True)

    scenario_defs, video_files = collect_video_files()
    if len(scenario_defs) == 0:
        print(f"No accessible videos found under {VIDEO_DIR}. Nothing to run.")
        return

    if demo_mode:
        print("Running in DEMO mode: skipping Cyton connection and EEG recording.")
    else:
        print("Running in LIVE mode: connecting to Cyton and recording EEG.")

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

    board = None
    if not demo_mode:
        board = start_cyton_stream()
        start_data_thread(board, stop_event, queue_in)
        eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID)
        aux_channels = board.get_analog_channels(CYTON_BOARD_ID)
    else:
        eeg_channels = []
        aux_channels = []

    eeg = np.zeros((len(eeg_channels), 0))
    aux = np.zeros((len(aux_channels), 0))
    timestamp = np.zeros((0,))
    markers = []

    def drain_queue():
        nonlocal eeg, aux, timestamp
        if demo_mode:
            return
        while not queue_in.empty():
            eeg_in, aux_in, ts_in = queue_in.get()
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            aux = np.concatenate((aux, aux_in), axis=1)
            timestamp = np.concatenate((timestamp, ts_in), axis=0)

    try:
        total_trials = len(trial_sequence)
        for i_trial, scenario_id in enumerate(trial_sequence, start=1):
            scenario = scenario_defs[scenario_id]
            if len(video_files[scenario_id]) == 0:
                print(
                    f"Skipping trial {i_trial}/{total_trials} - "
                    f"{scenario['category']} {scenario['scenario']}: no accessible videos left"
                )
                continue

            movie = None
            video_path = None
            while len(video_files[scenario_id]) > 0 and movie is None:
                candidate_video = random.choice(video_files[scenario_id])
                if not os.path.isfile(candidate_video) or not os.access(candidate_video, os.R_OK):
                    print(f"Skipping inaccessible video during run: {candidate_video}")
                    video_files[scenario_id].remove(candidate_video)
                    continue

                try:
                    movie = MovieStim(window, candidate_video, loop=False, noAudio=True)
                    video_path = candidate_video
                except Exception as exc:
                    print(f"Skipping unplayable video during run: {candidate_video} ({exc})")
                    video_files[scenario_id].remove(candidate_video)

            if movie is None or video_path is None:
                print(
                    f"Skipping trial {i_trial}/{total_trials} - "
                    f"{scenario['category']} {scenario['scenario']}: all videos unavailable"
                )
                continue

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
        if board is not None:
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
                "demo_mode": demo_mode,
                "categories": CATEGORIES,
                "scenarios": SCENARIOS,
            }, f, indent=2)

        print(f"Saved data to {DATA_DIR}")


if __name__ == "__main__":
    main()
