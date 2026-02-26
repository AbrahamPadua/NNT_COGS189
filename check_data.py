import numpy as np
import json
import os
import matplotlib.pyplot as plt

data_dir = r"c:\Users\abpadua\Desktop\NTT\NNT_COGS189\data\sub-01\ses-01\run-03"


def plot_channel_1_example():
    eeg_path = os.path.join(data_dir, "eeg.npy")
    ts_path = os.path.join(data_dir, "timestamp.npy")

    if not os.path.exists(eeg_path):
        print("[MISSING] eeg.npy (cannot generate channel 1 plot)")
        return

    try:
        eeg = np.load(eeg_path)
    except Exception as e:
        print(f"[ERROR] Could not load eeg.npy: {e}")
        return

    if eeg.ndim != 2 or eeg.shape[0] < 1 or eeg.shape[1] == 0:
        print(f"[ERROR] eeg.npy has unexpected shape for plotting: {eeg.shape}")
        return

    ch1 = eeg[0]
    x = np.arange(ch1.shape[0])
    x_label = "Sample index"

    if os.path.exists(ts_path):
        try:
            timestamp = np.load(ts_path)
            if timestamp.ndim == 1 and timestamp.shape[0] == ch1.shape[0]:
                x = timestamp - timestamp[0]
                x_label = "Time (s)"
        except Exception as e:
            print(f"[WARN] Could not use timestamp.npy for x-axis: {e}")

    plt.figure(figsize=(12, 4))
    plt.plot(x, ch1, linewidth=0.8)
    plt.title("EEG Channel 1")
    plt.xlabel(x_label)
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(data_dir, "channel1_example.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved channel 1 example plot: {out_path}")

def check_npy(filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"[MISSING] {filename}")
        return

    try:
        data = np.load(path)
        print(f"\n--- {filename} ---")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        if np.issubdtype(data.dtype, np.number):
            print(f"Min: {np.nanmin(data)}, Max: {np.nanmax(data)}")
            print(f"Has NaNs: {np.isnan(data).any()}")
            print(f"Has Infs: {np.isinf(data).any()}")
        else:
            print(f"Sample: {data[:5]}")
            
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")

def check_json(filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"[MISSING] {filename}")
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"\n--- {filename} ---")
        print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2))
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")

print(f"Checking data in: {data_dir}")
check_npy("eeg.npy")
check_npy("aux.npy")
check_npy("timestamp.npy")
check_json("markers.json")
check_json("metadata.json")
plot_channel_1_example()
