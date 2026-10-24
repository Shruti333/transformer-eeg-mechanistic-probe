import torch
import torch.nn as nn
from src.data_loader_physio import load_and_preprocess_physionet
from src.model import EEGTransformer
from src.activation_patching import ActivationPatcher

def main():
    # 1. Load benchmark PhysioNet motor imagery data from local EDF file
    dataset = load_and_preprocess_physionet(filepath="src/S001R04.edf")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Extract dimensions dynamically from the tensor dataset (shape: [batch, n_channels, n_times])
    sample_x, sample_y = dataset[0]
    n_channels = sample_x.shape[0]  # Should be 64
    n_times = sample_x.shape[1]     # Should be 321

    print(f"Initializing EEGTransformer with n_channels={n_channels}, n_times={n_times}...")

    # 2. Initialize the Transformer model matching the signature in model.py
    model = EEGTransformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=2,
        d_model=64,
        n_heads=4,
        n_layers=2
    )
    
    print("Model initialized successfully.")

    # 3. Pull a clean and corrupted sample for mechanistic probing
    data_iter = iter(train_loader)
    inputs, labels = next(data_iter)
    
    clean_input = inputs[0:1]     # Sample representing Clean run
    corrupted_input = inputs[1:2] # Sample representing Corrupted/Baseline run

    # 4. Instantiate the Patcher and run causal intervention
    patcher = ActivationPatcher(model)
    
    # Target layer name matching PyTorch's TransformerEncoder layer structure
    target_layer = "transformer_encoder.layers.0" 
    
    print(f"Running causal activation patching experiment on {target_layer}...")
    try:
        clean_logits, patched_logits, recovery_score = patcher.causal_patching_experiment(
            clean_input, corrupted_input, target_layer
        )
        print("\nCausal patching experiment completed successfully!")
        print(f"Clean Logits: {clean_logits}")
        print(f"Patched Logits: {patched_logits}")
        print(f"Recovery Softmax Scores: {recovery_score}")
    except Exception as e:
        print(f"Execution note on hook targeting: {e}")

if __name__ == "__main__":
    main()