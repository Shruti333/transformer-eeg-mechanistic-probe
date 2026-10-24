import mne
import numpy as np
import torch
from torch.utils.data import TensorDataset

def load_and_preprocess_physionet(filepath="src/S001R04.edf", *args, **kwargs):
    """
    Loads the manually downloaded PhysioNet EDF file directly from the local path,
    applies bandpass filtering (8-30 Hz mu/beta rhythms), epoching, and returns PyTorch Tensors.
    """
    print(f"Loading local PhysioNet file: {filepath}...")
    
    # Read the raw EDF file directly from disk
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    
    # Clean up channel names by stripping any extra whitespace
    raw.rename_channels(lambda x: x.strip())
    
    # Apply bandpass filter to capture mu and beta rhythms relevant to motor imagery (8-30 Hz)
    raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
    
    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    if len(events) == 0:
        # Fallback if annotations aren't structured as events: slice continuous data into fixed-size windows
        print("No annotations found. Slicing continuous data into fixed windows...")
        epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.2, preload=True, verbose=False)
        X = epochs.get_data()
        y = np.zeros(X.shape[0], dtype=int)
    else:
        # Create epochs (-0.5s to 1.5s around stimulus)
        tmin, tmax = -0.5, 1.5
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True, verbose=False)
        X = epochs.get_data()
        y = epochs.events[:, -1]
        y = y - y.min() # Normalize labels to 0-indexed

    # Normalize data per channel
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    print(f"Local dataset ready: X shape {X_tensor.shape}, y shape {y_tensor.shape}")
    return TensorDataset(X_tensor, y_tensor)

if __name__ == "__main__":
    load_and_preprocess_physionet()