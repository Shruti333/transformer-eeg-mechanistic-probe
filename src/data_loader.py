import os
import mne
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess_eeg(batch_size=32):
    """
    Downloads open-source MNE sample EEG data, preprocesses it, 
    and returns PyTorch DataLoaders for classification (Auditory vs. Visual stimuli).
    """
    print("[-] Fetching open-source MNE sample EEG dataset...")
    data_path = mne.datasets.sample.data_path()
    raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
    
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    
    # Keep only EEG channels to focus purely on electroencephalography
    raw.pick_types(eeg=True, stim=True, exclude='bads')
    
    # 1. Bandpass filter (1 Hz to 40 Hz) to isolate cognitive bands
    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    
    # 2. Extract events (Auditory Left/Right vs Visual Left/Right)
    events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
    
    # Define event IDs for classification task: 
    # Event 1/2 = Auditory (Class 0), Event 3/4 = Visual (Class 1)
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 
                'Visual/Left': 3, 'Visual/Right': 4}
    
    tmin, tmax = -0.2, 0.5  # Epoch window: 200ms pre-stimulus to 500ms post-stimulus
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), 
                        preload=True, verbose=False)
    
    # Extract data: shape -> (n_epochs, n_channels, n_times)
    X = epochs.get_data(copy=True)
    
    # Create binary labels: Auditory (0) vs Visual (1)
    # Map MNE sample event IDs: 1, 2 = Auditory (0); 3, 4 = Visual (1)
    y = np.array([0 if code in [1, 2] else 1 for code in epochs.events[:, 2]])
    
    # Standardize data across time per channel
    X_mean = X.mean(axis=2, keepdims=True)
    X_std = X.std(axis=2, keepdims=True)
    X = (X - X_mean) / (X_std + 1e-8)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Train/Test Split (80/20)
    dataset_size = len(X_tensor)
    train_size = int(0.8 * dataset_size)
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
    test_dataset = TensorDataset(X_tensor[test_indices], y_tensor[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[+] Data loaded successfully! Channels: {X.shape[1]}, Time samples: {X.shape[2]}")
    print(f"[+] Training samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    return train_loader, test_loader, epochs.ch_names
