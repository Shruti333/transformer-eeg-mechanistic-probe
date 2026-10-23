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
