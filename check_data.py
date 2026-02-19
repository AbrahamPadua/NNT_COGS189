import numpy as np
import json
import os

data_dir = r"c:\Users\abpadua\Desktop\NTT\NNT_COGS189\data\sub-01\ses-01\run-01"

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
