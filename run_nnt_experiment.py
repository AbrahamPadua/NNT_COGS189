import os
import sys
import time
import json
import glob
import random
import hashlib
import subprocess
from queue import Queue
from threading import Thread, Event

import numpy as np
from psychopy import visual, core
from psychopy.hardware import keyboard
from psychopy.visual.movie import MovieStim
from serial import Serial
import serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams

try:
    import cv2
except Exception:
    cv2 = None

try:
    import imageio
except Exception:
    imageio = None

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
# TRIALS_PER_SCENARIO = 100
TRIALS_PER_SCENARIO = 3
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
PREFERRED_VIDEO_BACKEND = "imageio"  # Options: "imageio", "opencv", "psychopy"

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

def resolve_ffmpeg_exe():
    """Resolve an ffmpeg executable path for optional video transcoding fallback."""
    ffmpeg_from_env = os.environ.get("FFMPEG_EXE")
    if ffmpeg_from_env and os.path.isfile(ffmpeg_from_env):
        return ffmpeg_from_env

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def transcode_video_for_psychopy(input_video_path, cache_dir, ffmpeg_exe):
    """Create a PsychoPy-friendly MP4 by stripping metadata and normalizing encoding."""
    if ffmpeg_exe is None:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    file_hash = hashlib.sha1(input_video_path.encode("utf-8")).hexdigest()[:12]
    input_base = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = os.path.join(cache_dir, f"{input_base}_{file_hash}_clean.mp4")

    if os.path.isfile(output_video_path) and os.path.getsize(output_video_path) > 0:
        return output_video_path

    command = [
        ffmpeg_exe,
        "-y",
        "-i", input_video_path,
        "-map_metadata", "-1",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_video_path,
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        print(f"Failed to launch ffmpeg for transcoding {input_video_path}: {exc}")
        return None

    if result.returncode != 0 or not os.path.isfile(output_video_path):
        stderr_tail = (result.stderr or "")[-1000:]
        print(
            f"Failed to transcode {input_video_path} for PsychoPy. "
            f"ffmpeg exit code: {result.returncode}. Details: {stderr_tail}"
        )
        return None

    return output_video_path


def can_decode_with_opencv(video_path):
    """Check if OpenCV can decode at least one frame from a video."""
    if cv2 is None:
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False

    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None)


def can_decode_with_imageio(video_path):
    """Check if imageio-ffmpeg can decode at least one frame from a video."""
    if imageio is None:
        return False

    reader = None
    try:
        reader = imageio.get_reader(video_path)
        frame = reader.get_data(0)
        return frame is not None
    except Exception:
        return False
    finally:
        if reader is not None:
            reader.close()


def to_psychopy_image_array(frame_rgb):
    """Convert frame data to a strict numpy.ndarray in PsychoPy's [-1, 1] range."""
    frame_rgb_np = np.array(frame_rgb, dtype=np.float32, copy=True)
    return np.ascontiguousarray(frame_rgb_np / 127.5 - 1.0)


def play_video_opencv(window, kb, video_path, duration):
    """Play video using OpenCV decode + PsychoPy ImageStim draw fallback."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not available for video playback fallback.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    image_stim = None
    stim_size = None
    stim_clock = core.Clock()

    try:
        while stim_clock.getTime() < duration:
            if 'escape' in kb.getKeys():
                raise KeyboardInterrupt

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb_psychopy = to_psychopy_image_array(frame_rgb)

            if stim_size is None:
                frame_h, frame_w = frame_rgb.shape[:2]
                win_w, win_h = window.size
                scale = min(win_w / frame_w, win_h / frame_h)
                stim_size = (int(frame_w * scale), int(frame_h * scale))

            if image_stim is None:
                image_stim = visual.ImageStim(
                    window,
                    image=frame_rgb_psychopy,
                    size=stim_size,
                    units='pix',
                    interpolate=True,
                )
            else:
                image_stim.image = frame_rgb_psychopy

            image_stim.draw()
            window.flip()
    finally:
        cap.release()


def play_video_imageio(window, kb, video_path, duration):
    """Play video using imageio (ffmpeg) decode + PsychoPy ImageStim draw."""
    if imageio is None:
        raise RuntimeError("imageio is not available for video playback.")

    imageio_lib = imageio

    def open_reader(path):
        reader_local = imageio_lib.get_reader(path)
        metadata = reader_local.get_meta_data() or {}
        fps = float(metadata.get('fps') or 24.0)
        if fps <= 0:
            fps = 24.0
        return reader_local, fps

    reader = None
    image_stim = None
    stim_size = None
    stim_clock = core.Clock()

    try:
        reader, fps = open_reader(video_path)
        frame_interval = 1.0 / fps
        next_frame_time = 0.0
        frame_index = 0

        while stim_clock.getTime() < duration:
            if 'escape' in kb.getKeys():
                raise KeyboardInterrupt

            now = stim_clock.getTime()
            if now < next_frame_time:
                core.wait(min(0.002, next_frame_time - now))
                continue

            try:
                frame_rgb = reader.get_data(frame_index)
                frame_index += 1
            except Exception:
                reader.close()
                reader, fps = open_reader(video_path)
                frame_interval = 1.0 / fps
                frame_index = 0
                frame_rgb = reader.get_data(frame_index)
                frame_index += 1

            if frame_rgb is None:
                continue

            frame_rgb_psychopy = to_psychopy_image_array(frame_rgb)

            if stim_size is None:
                frame_h, frame_w = frame_rgb.shape[:2]
                win_w, win_h = window.size
                scale = min(win_w / frame_w, win_h / frame_h)
                stim_size = (int(frame_w * scale), int(frame_h * scale))

            if image_stim is None:
                image_stim = visual.ImageStim(
                    window,
                    image=frame_rgb_psychopy,
                    size=stim_size,
                    units='pix',
                    interpolate=True,
                )
            else:
                image_stim.image = frame_rgb_psychopy

            image_stim.draw()
            window.flip()
            next_frame_time += frame_interval
    finally:
        if reader is not None:
            reader.close()

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
    ffmpeg_exe = resolve_ffmpeg_exe()
    transcoded_cache_dir = os.path.join(DATA_DIR, "video_cache")

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
            playback_backend = None
            use_imageio_fallback = False
            use_opencv_fallback = False
            while (
                len(video_files[scenario_id]) > 0
                and movie is None
                and not use_imageio_fallback
                and not use_opencv_fallback
            ):
                candidate_video = random.choice(video_files[scenario_id])
                if not os.path.isfile(candidate_video) or not os.access(candidate_video, os.R_OK):
                    print(f"Skipping inaccessible video during run: {candidate_video}")
                    video_files[scenario_id].remove(candidate_video)
                    continue

                if PREFERRED_VIDEO_BACKEND == "imageio" and can_decode_with_imageio(candidate_video):
                    use_imageio_fallback = True
                    video_path = candidate_video
                    playback_backend = "imageio"
                    print(f"Using ImageIO backend for video: {candidate_video}")
                    continue

                if PREFERRED_VIDEO_BACKEND == "opencv" and can_decode_with_opencv(candidate_video):
                    use_opencv_fallback = True
                    video_path = candidate_video
                    playback_backend = "opencv"
                    print(f"Using OpenCV backend for video: {candidate_video}")
                    continue

                try:
                    movie = MovieStim(window, candidate_video, loop=False, noAudio=True)
                    video_path = candidate_video
                    playback_backend = "psychopy"
                except Exception as exc:
                    print(f"Skipping unplayable video during run: {candidate_video} ({exc})")

                    transcoded_path = transcode_video_for_psychopy(
                        candidate_video,
                        transcoded_cache_dir,
                        ffmpeg_exe,
                    )

                    if transcoded_path is not None:
                        try:
                            movie = MovieStim(window, transcoded_path, loop=False, noAudio=True)
                            video_path = transcoded_path
                            playback_backend = "psychopy"
                            print(
                                f"Recovered video by transcoding: {candidate_video} -> {transcoded_path}"
                            )
                            continue
                        except Exception as transcode_exc:
                            print(
                                "Transcoded video still unplayable: "
                                f"{transcoded_path} ({transcode_exc})"
                            )

                    fallback_video = transcoded_path if transcoded_path is not None else candidate_video
                    if can_decode_with_imageio(fallback_video):
                        use_imageio_fallback = True
                        video_path = fallback_video
                        playback_backend = "imageio"
                        print(
                            f"Recovered video with ImageIO backend: {candidate_video} -> {fallback_video}"
                        )
                        continue

                    if can_decode_with_opencv(fallback_video):
                        use_opencv_fallback = True
                        video_path = fallback_video
                        playback_backend = "opencv"
                        print(
                            f"Recovered video with OpenCV fallback: {candidate_video} -> {fallback_video}"
                        )
                        continue

                    video_files[scenario_id].remove(candidate_video)

            if (movie is None and not use_imageio_fallback and not use_opencv_fallback) or video_path is None:
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
                "video_backend": playback_backend,
                "stim_start_time": time.time(),
                "start_sample_index": int(eeg.shape[1]),
            }

            if playback_backend == "imageio":
                play_video_imageio(window, kb, video_path, STIM_DURATION)
            elif playback_backend == "opencv":
                play_video_opencv(window, kb, video_path, STIM_DURATION)
            else:
                if movie is None:
                    print(
                        f"Skipping trial {i_trial}/{total_trials} - "
                        f"{scenario['category']} {scenario['scenario']}: video backend setup failed"
                    )
                    continue

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
