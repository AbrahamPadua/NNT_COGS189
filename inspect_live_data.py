import argparse
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
import serial
from serial.tools import list_ports


CYTON_BOARD_ID = 0  # 0 if no daisy, 2 if daisy, 6 if daisy+wifi
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'


def find_openbci_port():
    """Find the serial port for an OpenBCI Cyton dongle on the current machine."""
    candidates = []
    for port in list_ports.comports():
        info = f"{port.device} {port.description} {port.manufacturer}"
        if any(k in info.lower() for k in ["openbci", "cyton", "usb serial", "cp210", "ch340", "ftdi"]):
            candidates.append(port.device)

    if not candidates:
        if sys.platform.startswith('win'):
            candidates = [f'COM{i + 1}' for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            candidates = []
        elif sys.platform.startswith('darwin'):
            candidates = []

    for port in candidates:
        try:
            s = Serial(port=port, baudrate=BAUD_RATE, timeout=1)
            s.write(b'v')
            line = ''
            time.sleep(1.5)
            while s.in_waiting and '$$$' not in line:
                line += s.read().decode('utf-8', errors='replace')
            s.close()
            if 'OpenBCI' in line:
                return port
        except (OSError, serial.SerialException):
            continue

    raise OSError('Cannot find OpenBCI port.')


def start_board():
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


def monitor_live_data(duration_sec=0, display_sec=5.0, refresh_hz=20.0, flatline_std_uv=1.0):
    board = start_board()
    eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID)
    sampling_rate = BoardShim.get_sampling_rate(CYTON_BOARD_ID)

    window_samples = max(1, int(display_sec * sampling_rate))
    refresh_dt = 1.0 / max(1.0, refresh_hz)

    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.linspace(-display_sec, 0, window_samples)

    lines = []
    offsets = np.arange(len(eeg_channels))[::-1] * 250.0
    for i in range(len(eeg_channels)):
        line, = ax.plot(x, np.zeros_like(x) + offsets[i], linewidth=1.0)
        lines.append(line)

    ax.set_title('Live EEG Monitor (Cyton)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude + channel offset (uV)')
    ax.set_xlim(-display_sec, 0)
    ax.set_ylim(-250, max(offsets) + 250)
    ax.grid(True, alpha=0.3)

    channel_labels = [f"CH{idx + 1}" for idx in range(len(eeg_channels))]
    for i, label in enumerate(channel_labels):
        ax.text(-display_sec + 0.05, offsets[i] + 120, label, fontsize=8)

    status_text = ax.text(
        0.01,
        0.99,
        '',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    print('Streaming started. Close plot window or press Ctrl+C to stop.')
    t0 = time.time()

    try:
        while True:
            if duration_sec > 0 and (time.time() - t0) >= duration_sec:
                break

            data = board.get_current_board_data(window_samples)
            if data.size == 0 or data.shape[1] < 2:
                plt.pause(refresh_dt)
                continue

            eeg = data[eeg_channels, :]
            if eeg.shape[1] < window_samples:
                pad = np.tile(eeg[:, :1], (1, window_samples - eeg.shape[1]))
                eeg = np.concatenate([pad, eeg], axis=1)

            channel_std = np.std(eeg, axis=1)
            flat_mask = channel_std < flatline_std_uv

            for i, line in enumerate(lines):
                demeaned = eeg[i] - np.mean(eeg[i])
                line.set_ydata(demeaned + offsets[i])
                line.set_color('red' if flat_mask[i] else 'tab:blue')

            flat_channels = [str(i + 1) for i, is_flat in enumerate(flat_mask) if is_flat]
            flat_msg = 'None' if len(flat_channels) == 0 else ', '.join(flat_channels)

            status_text.set_text(
                f"Sampling rate: {sampling_rate} Hz\n"
                f"Display window: {display_sec:.1f} s\n"
                f"Flatline threshold (std): {flatline_std_uv:.2f} uV\n"
                f"Flat channels: {flat_msg}"
            )

            fig.canvas.draw_idle()
            plt.pause(refresh_dt)

    except KeyboardInterrupt:
        pass
    finally:
        board.stop_stream()
        board.release_session()
        plt.ioff()
        plt.show()
        print('Live monitor stopped.')


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect live Cyton EEG data in real time.')
    parser.add_argument('--duration', type=float, default=0.0, help='Seconds to run (0 = until interrupted).')
    parser.add_argument('--display-sec', type=float, default=5.0, help='Seconds shown in rolling plot window.')
    parser.add_argument('--refresh-hz', type=float, default=20.0, help='Plot update rate.')
    parser.add_argument('--flatline-std', type=float, default=1.0, help='Std threshold (uV) for flatline warning.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    monitor_live_data(
        duration_sec=args.duration,
        display_sec=args.display_sec,
        refresh_hz=args.refresh_hz,
        flatline_std_uv=args.flatline_std,
    )
