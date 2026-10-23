import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import load_and_preprocess_eeg
from src.model import EEGTransformer
from src.interpretability import extract_attention_weights
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Advanced NeuroAI Transformer Pipeline on: {device}")
    
    # 1. Load Data
    train_loader, test_loader, ch_names = load_and_preprocess_eeg(batch_size=32)
    sample_x, _ = next(iter(train_loader))
    n_channels, n_times = sample_x.shape[1], sample_x.shape[2]
    
    # 2. Initialize Transformer Model
    model = EEGTransformer(n_channels=n_channels, n_times=n_times, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    
    # 3. Training Loop
    epochs = 8
    print("\n[-] Training Transformer EEG Decoder...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.2f}%")
        
    # 4. Evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    print(f"\n[+] Final Test Decoding Accuracy: {test_correct/test_total*100:.2f}%")
        
    # 5. Attention Interpretability Analysis
    attn_maps = extract_attention_weights(model, test_loader)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(attn_maps[0], cmap='magma', aspect='auto')
    plt.colorbar(label='Attention Weight Intensity')
    plt.title('Mechanistic Interpretability: Transformer Attention Head 0')
    plt.xlabel('Key Time Step')
    plt.ylabel('Query Time Step')
    plt.savefig('transformer_attention_probe.png', dpi=300, bbox_inches='tight')
    print("[+] Saved advanced attention interpretability map to 'transformer_attention_probe.png'")

if __name__ == '__main__':
    main()
