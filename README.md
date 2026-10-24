# Transformer EEG Mechanistic Probe (`transformer-eeg-mechanistic-probe`)

A repository combining self-supervised multi-head attention Transformers with electroencephalography (EEG) signal processing and mechanistic interpretability for brain-computer interface (BCI) decoding.

## Overview
This repository implements an end-to-end research pipeline:
1. **Electrophysiological Data Preprocessing:** Pulls benchmark MNE EEG recordings, applies bandpass filtering (1–40 Hz), and epochs multi-channel time series around auditory vs. visual sensory processing tasks.
2. **Transformer Architecture (`EEGTransformer`):** Implements a temporal-spatial Multi-Head Self-Attention Transformer Encoder to capture long-range contextual dependencies across brain oscillations.
3. **Mechanistic Interpretability:** Extracts and visualizes internal self-attention head dynamics to probe how artificial attention weights map to biological temporal processing windows.

## Installation & Execution

1. Extract the ZIP file and navigate into the project directory:
   ```bash
   cd transformer-eeg-mechanistic-probe
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute the full pipeline:
   ```bash
   python main.py
   ```
---

## Advanced Mechanistic Probing Update (`main_2.py`)

This repository has been expanded to include advanced mechanistic interpretability pipelines:

1. **PhysioNet Benchmark Integration (`data_loader_physio.py`):** Programmatically fetches 64-channel PhysioNet motor imagery EEG datasets, applying bandpass filtering (8–30 Hz Mu/Beta rhythms) and custom epoching.
2. **Causal Activation Patching (`activation_patching.py`):** Implements Neel Nanda-style causal interventions using PyTorch forward hooks to swap and test intermediate attention head activations, measuring exact logit recovery and circuit necessity.
3. **Sparse Autoencoders (SAEs):** Disentangles polysemantic representations within the transformer residual stream into monosemantic features.

### Execution for Advanced Pipeline
To run the new PhysioNet loader and causal activation patching experiment:
```bash
python main_2.py

---


 **Sparse Autoencoders for Monosemantic Probing (`sparse_autoencoder.py`):**
   * Decomposes polysemantic activations from the transformer's residual stream into sparse, interpretable latent features using an expansion factor and an $L_1$ sparsity penalty.
   * Enforces unit-norm constraints on decoder weights to maintain stable feature dictionaries.

 **Causal Activation Patching (`activation_patching.py` & `main_3.py`):**
   * Utilizes PyTorch forward hooks to perform causal interventions (swapping intermediate layer states between clean and corrupted runs).
   * Quantifies exact circuit necessity and logit recovery scores across multi-class motor imagery states.

---

### 💻 Execution & Workflow

To run the complete pipeline—including model training, SAE feature extraction, and causal intervention patching—execute:

```bash
python main_3.py