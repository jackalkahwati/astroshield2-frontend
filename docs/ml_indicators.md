# ML Indicators Documentation

This document provides detailed information about the ML-based indicators used in the AstroShield CCDM microservice.

## 1. Maneuver Detection

### Model Architecture
- **Type**: LSTM Autoencoder
- **Purpose**: Unsupervised detection of trajectory anomalies
- **Input**: Time series of position, velocity, and acceleration data
- **Output**: Reconstruction error-based anomaly scores

### Indicators

#### 1.1 Subtle Maneuver Detection
- **Indicator Name**: `subtle_maneuver_detected`
- **Threshold**: 0.3
- **Features**:
  * Position time series
  * Velocity changes
  * Acceleration patterns
- **Purpose**: Detect small, subtle changes in trajectory that might indicate covert maneuvers
- **Detection Method**: Reconstruction error analysis with lower threshold

#### 1.2 Significant Maneuver Detection
- **Indicator Name**: `significant_maneuver_detected`
- **Threshold**: 0.5
- **Features**:
  * Position time series
  * Velocity changes
  * Acceleration patterns
- **Purpose**: Detect major trajectory changes indicating clear maneuvers
- **Detection Method**: Reconstruction error analysis with higher threshold

## 2. Signature Analysis

### Model Architecture
- **Type**: Convolutional Autoencoder
- **Purpose**: Unsupervised detection of signature anomalies
- **Input**: Multi-dimensional sensor data (optical and radar)
- **Output**: Reconstruction error-based anomaly scores

### Indicators

#### 2.1 Anomalous Signature
- **Indicator Name**: `anomalous_signature`
- **Threshold**: 0.4
- **Features**:
  * Optical signature patterns
  * Radar cross-section data
  * Combined sensor features
- **Purpose**: Detect unusual or unexpected signature characteristics
- **Detection Method**: Pattern reconstruction error analysis

#### 2.2 Signature Mismatch
- **Indicator Name**: `signature_mismatch`
- **Threshold**: 0.4
- **Features**:
  * Cross-sensor correlations
  * Temporal signature alignment
  * Feature consistency metrics
- **Purpose**: Detect inconsistencies between different sensor data
- **Detection Method**: Cross-sensor correlation analysis

## 3. AMR Analysis

### Model Architecture
- **Type**: Variational Autoencoder
- **Purpose**: Unsupervised detection of AMR anomalies
- **Input**: AMR measurements and historical data
- **Output**: Probabilistic anomaly scores

### Indicators

#### 3.1 AMR Anomaly
- **Indicator Name**: `amr_anomaly`
- **Threshold**: 0.4
- **Features**:
  * Current AMR measurements
  * Historical AMR patterns
  * Environmental factors
- **Purpose**: Detect unusual AMR characteristics
- **Detection Method**: Latent space anomaly detection

#### 3.2 AMR Change
- **Indicator Name**: `amr_change`
- **Threshold**: 20% deviation from predicted
- **Features**:
  * Current AMR value
  * Predicted AMR value
  * Historical trends
- **Purpose**: Detect significant changes in AMR over time
- **Detection Method**: Comparison with VAE predictions

## Risk Assessment

The overall risk assessment combines these indicators using the following weights:

1. **Maneuver Risk**:
   - Calculated from highest confidence level of maneuver indicators
   - Critical if > 0.8
   - High if > 0.6
   - Moderate if > 0.4
   - Low otherwise

2. **Signature Risk**:
   - Calculated from highest confidence level of signature indicators
   - Same thresholds as maneuver risk

3. **AMR Risk**:
   - Calculated from highest confidence level of AMR indicators
   - Same thresholds as maneuver risk

The final risk level is determined by the highest risk score across all categories.
