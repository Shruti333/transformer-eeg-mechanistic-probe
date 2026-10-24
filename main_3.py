import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loader_physio import load_and_preprocess_physionet
from src.model import EEGTransformer
from src.activation_patching import ActivationPatcher
from src.sparse_autoencoder import SparseAutoencoder

def train_transformer(model, dataloader, epochs=5):
    """Trains the EEGTransformer on motor imagery classification."""
    print("\n--- Training EEGTransformer ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc:.2f}%")

def train_sae_on_activations(activations, d_model=64, epochs=50):
    """Trains a Sparse Autoencoder on the transformer's residual stream."""
    print("\n--- Training Sparse Autoencoder (SAE) ---")
    sae = SparseAutoencoder(d_model=d_model, expansion_factor=4)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    
    sae.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, h = sae(activations)
        loss, recon_loss, sparsity_loss = sae.compute_loss(activations, recon, h, l1_coefficient=1e-3)
        loss.backward()
        optimizer.step()
        
        # Enforce unit norm constraint on decoder weights
        with torch.no_grad():
            sae.W_dec.data = nn.functional.normalize(sae.W_dec.data, dim=0)
            
        if (epoch + 1) % 10 == 0:
            print(f"SAE Epoch [{epoch+1}/{epochs}] | Total Loss: {loss.item():.4f} | Recon Loss: {recon_loss.item():.4f} | Sparsity: {sparsity_loss.item():.4f}")
            
    return sae

def main():
    # 1. Load PhysioNet Data
    dataset = load_and_preprocess_physionet(filepath="src/S001R04.edf")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    sample_x, sample_y = dataset[0]
    n_channels = sample_x.shape[0]
    n_times = sample_x.shape[1]

    # Dynamically detect unique classes to prevent label out-of-bounds errors
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    n_classes = len(set(all_labels))
    print(f"Detected {n_classes} unique class label(s) in dataset: {set(all_labels)}")

    # 2. Initialize Model with dynamic n_classes
    model = EEGTransformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        d_model=64,
        n_heads=4,
        n_layers=2
    )

    # 3. Train the Transformer
    train_transformer(model, train_loader, epochs=5)

    # 4. Extract Residual Stream Activations for SAE
    print("\n--- Extracting Residual Stream Activations ---")
    model.eval()
    with torch.no_grad():
        all_inputs, _ = next(iter(train_loader))
        x = model.channel_projection(all_inputs.transpose(1, 2)) + model.pos_embedding[:, :all_inputs.size(2), :]
        residual_stream_activations = model.transformer_encoder(x)

    # 5. Train Sparse Autoencoder on the Activations
    sae = train_sae_on_activations(residual_stream_activations, d_model=64, epochs=30)

    # 6. Run Causal Activation Patching on the Trained Model
    print("\n--- Running Causal Activation Patching ---")
    patcher = ActivationPatcher(model)
    clean_input = all_inputs[0:1]
    corrupted_input = all_inputs[1:2]
    target_layer = "transformer_encoder.layers.0"

    clean_logits, patched_logits, recovery_score = patcher.causal_patching_experiment(
        clean_input, corrupted_input, target_layer
    )
    
    print("Causal Patching Pipeline Completed Successfully!")
    print(f"Clean Logits: {clean_logits}")
    print(f"Patched Logits: {patched_logits}")
    print(f"Recovery Scores: {recovery_score}")

if __name__ == "__main__":
    main()