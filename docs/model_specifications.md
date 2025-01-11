# AstroShield Model Specifications

## Model Overview
This document provides specifications for the machine learning models used in the AstroShield CCDM Microservice.

### 1. Stability LSTM Model
- **Purpose**: Evaluate orbital stability and detect anomalies
- **Architecture**: LSTM with Attention Mechanism
- **Input Format**: 
  - Shape: (batch_size, 60, 6)
  - Features: [x, y, z, vx, vy, vz]
- **Output Format**:
  - Shape: (batch_size, 3)
  - Values: [stability_score, family_deviation, anomaly_score]
- **Thresholds**:
  - Stability: 0.8
  - Family Deviation: 2.0σ
  - Anomaly: 0.9
- **Files**:
  - PyTorch: `models/stability_model.pth`
  - ONNX: `models/stability_model.onnx`

### 2. Maneuver LSTM Autoencoder
- **Purpose**: Detect spacecraft maneuvers and analyze pattern of life
- **Architecture**: LSTM-based Autoencoder
- **Input Format**:
  - Shape: (batch_size, 60, 6)
  - Features: [x, y, z, vx, vy, vz]
- **Output Format**:
  - Shape: (batch_size, 60, 6)
  - Reconstructed trajectory
- **Thresholds**:
  - Reconstruction: 0.1
  - Maneuver Confidence: 0.95
- **Files**:
  - PyTorch: `models/maneuver_model.pth`
  - ONNX: `models/maneuver_model.onnx`

### 3. Signature CNN Autoencoder
- **Purpose**: Analyze multi-channel signature data
- **Architecture**: CNN-based Autoencoder
- **Input Format**:
  - Shape: (batch_size, 3, 64, 64)
  - Channels: Multi-spectral signature data
- **Output Format**:
  - Shape: (batch_size, 3, 64, 64)
  - Reconstructed signature
- **Threshold**:
  - Signature Confidence: 0.90
- **Files**:
  - PyTorch: `models/signature_model.pth`
  - ONNX: `models/signature_model.onnx`

### 4. Physical VAE
- **Purpose**: Analyze physical characteristics and detect sub-satellites
- **Architecture**: Variational Autoencoder
- **Input Format**:
  - Shape: (batch_size, 10)
  - Features: Physical characteristics
- **Output Format**:
  - Shape: (batch_size, 10)
  - Reconstructed characteristics
- **Threshold**:
  - AMR Deviation: 15%
- **Files**:
  - PyTorch: `models/physical_model.pth`
  - ONNX: `models/physical_model.onnx`

## Model Performance Metrics
- Training Loss Convergence: < 0.01
- Inference Time: < 1s per sample
- Memory Usage: < 100MB per model

## Version Information
- Model Version: 1.0.0
- Last Updated: 2024-03-19
- Compatible ONNX Version: 1.17.0
- PyTorch Version: 2.5.1

## Usage Notes
1. All models support dynamic batch sizes
2. LSTM models support variable sequence lengths
3. Models are optimized for CPU inference
4. Input data should be normalized to [-1, 1] range

## Security Considerations
- Models trained on synthetic data
- No external dependencies in inference
- Unsupervised learning approach
- Built-in anomaly detection
