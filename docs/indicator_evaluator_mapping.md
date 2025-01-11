# AstroShield CCDM Indicator-Evaluator Mapping

## Complete Indicator-Evaluator Matrix

| Indicator | Primary Evaluator | Support Evaluator | ML Enhancement |
|-----------|------------------|-------------------|----------------|
| Object is stable | StabilityEvaluator | - | - |
| Stability has changed | StabilityEvaluator | - | - |
| Maneuvers detected | ManeuverEvaluator | - | LSTM Autoencoder |
| RF detected | SignatureEvaluator | - | CNN Autoencoder |
| Sub-satellites deployed | PhysicalCharacteristicsEvaluator | - | VAE |
| Maneuvers/RF POL out of family | ManeuverEvaluator | SignatureEvaluator | LSTM Autoencoder |
| Violates ITU/FCC filings | ComplianceEvaluator | - | - |
| Class disagreement between analysts | AnalystEvaluator | - | - |
| Orbit out of family | StabilityEvaluator | - | - |
| Optical/RADAR signature out of family | SignatureEvaluator | - | CNN Autoencoder |
| Optical/RADAR signature mismatch | SignatureEvaluator | - | CNN Autoencoder |
| Stimulated by US/allied systems | StimulationEvaluator | SignatureEvaluator | - |
| AMR out of family | PhysicalCharacteristicsEvaluator | - | VAE |
| Notable AMR changes | PhysicalCharacteristicsEvaluator | - | VAE |
| Valid remote-sensing passes | BehaviorEvaluator | ManeuverEvaluator | LSTM Autoencoder |
| Imaging maneuvers detected | ManeuverEvaluator | BehaviorEvaluator | LSTM Autoencoder |
| Imaging maneuvers are POL violations | ManeuverEvaluator | BehaviorEvaluator | LSTM Autoencoder |
| Maneuvers in sensor gaps | ManeuverEvaluator | - | LSTM Autoencoder |
| Objects tracked > expected | LaunchEvaluator | - | - |
| Launch site/vehicle known threats | LaunchEvaluator | - | - |
| UNK/DEB SMA > parent | LaunchEvaluator | PhysicalCharacteristicsEvaluator | - |
| UCT during eclipse | EnvironmentalEvaluator | - | - |
| Relatively unoccupied orbit | EnvironmentalEvaluator | - | - |
| High radiation environment | EnvironmentalEvaluator | - | - |
| Not in UN registry | ComplianceEvaluator | - | - |

## Evaluator Summary

### Primary Evaluators
1. **StabilityEvaluator**
   - Stability assessment
   - Orbit family analysis
   - Orbital parameter tracking

2. **ManeuverEvaluator**
   - Maneuver detection
   - POL analysis
   - Sensor gap analysis
   - ML Enhanced: LSTM Autoencoder

3. **SignatureEvaluator**
   - RF detection
   - Optical/RADAR analysis
   - Cross-sensor correlation
   - ML Enhanced: CNN Autoencoder

4. **PhysicalCharacteristicsEvaluator**
   - Sub-satellite detection
   - AMR analysis
   - Physical change tracking
   - ML Enhanced: VAE

5. **LaunchEvaluator**
   - Object tracking
   - Launch site assessment
   - Debris analysis

6. **EnvironmentalEvaluator**
   - Eclipse tracking
   - Orbit environment
   - Radiation assessment

7. **ComplianceEvaluator**
   - Regulatory compliance
   - Registry verification

8. **AnalystEvaluator**
   - Classification consensus
   - Disagreement tracking

9. **StimulationEvaluator**
   - System interaction
   - Response analysis

10. **BehaviorEvaluator**
    - Remote sensing validation
    - Proximity analysis
    - Intent evaluation

## ML Model Application

### LSTM Autoencoder
- Maneuver detection
- POL analysis
- Sensor gap correlation
- Remote sensing validation

### CNN Autoencoder
- Signature analysis
- Cross-sensor correlation
- RF pattern detection

### Variational Autoencoder
- AMR analysis
- Physical change detection
- Sub-satellite verification

## Indicator Categories

1. **Stability Indicators** (3)
   - Object stability
   - Stability changes
   - Orbit family analysis

2. **Maneuver Indicators** (4)
   - Basic maneuvers
   - POL violations
   - Sensor gap maneuvers
   - Imaging maneuvers

3. **Signature Indicators** (3)
   - RF emissions
   - Optical/RADAR signatures
   - Cross-sensor correlation

4. **Physical Indicators** (3)
   - Sub-satellites
   - AMR characteristics
   - Physical changes

5. **Launch Indicators** (3)
   - Object count
   - Launch source
   - Debris analysis

6. **Environmental Indicators** (3)
   - Eclipse behavior
   - Orbit occupancy
   - Radiation environment

7. **Compliance Indicators** (2)
   - Regulatory compliance
   - Registry status

8. **Analyst Indicators** (1)
   - Classification consensus

9. **Stimulation Indicators** (1)
   - System interaction

10. **Behavior Indicators** (2)
    - Remote sensing
    - Proximity operations

## Detailed Indicator-Evaluator Documentation

### 1. Evaluator Details

#### StabilityEvaluator
**Purpose**: Assess object stability and orbital characteristics
**Key Features**:
- Orbital parameter tracking (SMA, eccentricity, inclination)
- Historical stability analysis
- Family comparison algorithms
**Thresholds**:
- Stability Change: >3σ from historical mean
- Orbit Family: >2σ from documented parameters
**Data Sources**:
- TLE history
- Ephemeris data
- Historical maneuver catalog

#### ManeuverEvaluator (ML Enhanced)
**Purpose**: Detect and classify spacecraft maneuvers
**Key Features**:
- Delta-V calculation
- Burn duration estimation
- POL analysis
**Thresholds**:
- Maneuver Detection: >1m/s Delta-V
- POL Violation: >95% confidence
- Sensor Gap: >80% correlation
**ML Integration**:
- LSTM for temporal pattern recognition
- Sequence prediction for gap analysis
- Anomaly scoring for POL

#### SignatureEvaluator (ML Enhanced)
**Purpose**: Analyze RF, optical, and radar signatures
**Key Features**:
- Cross-sensor fusion
- Frequency analysis
- Signature change detection
**Thresholds**:
- RF Detection: >-120 dBm
- Signature Mismatch: >90% confidence
- Family Analysis: >2σ deviation
**ML Integration**:
- CNN for signature classification
- Feature extraction from raw signals
- Cross-sensor correlation

#### PhysicalCharacteristicsEvaluator (ML Enhanced)
**Purpose**: Track physical object changes
**Key Features**:
- AMR calculation
- Size estimation
- Material properties
**Thresholds**:
- AMR Change: >15% deviation
- Size Change: >20% variation
- Sub-satellite: >95% confidence
**ML Integration**:
- VAE for characteristic modeling
- Anomaly detection in physical parameters
- Change point detection

#### LaunchEvaluator
**Purpose**: Analyze launch events and object relationships
**Key Features**:
- Object counting
- Parent-child relationships
- Launch site correlation
**Thresholds**:
- Object Count: >expected+1
- SMA Difference: >1km from parent
- Threat Source: >90% confidence

#### EnvironmentalEvaluator
**Purpose**: Assess space environment context
**Key Features**:
- Eclipse prediction
- Radiation modeling
- Orbital regime analysis
**Thresholds**:
- UCT Correlation: >95% in eclipse
- Orbit Occupancy: <5 objects/degree
- Radiation: >typical+2σ

### 2. ML Model Integration Details

#### LSTM Autoencoder for Maneuver Detection
**Architecture**:
- Input: 60-day position/velocity sequence
- Encoder: 3 LSTM layers (128, 64, 32 units)
- Decoder: 3 LSTM layers (32, 64, 128 units)
- Output: Reconstructed sequence

**Training**:
- Loss: MSE + KL divergence
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 100

**Application**:
1. Maneuver Detection:
   - Reconstruction error thresholding
   - Temporal anomaly scoring
   - Pattern deviation analysis

2. POL Analysis:
   - Sequence prediction
   - Historical pattern matching
   - Confidence scoring

3. Sensor Gap Analysis:
   - Trajectory prediction
   - Gap behavior modeling
   - Anomaly detection

#### CNN Autoencoder for Signature Analysis
**Architecture**:
- Input: Multi-channel signature data
- Encoder: 4 Conv2D layers
- Latent space: 128 dimensions
- Decoder: 4 TransposeConv2D layers

**Training**:
- Loss: Binary cross-entropy
- Optimizer: RMSprop
- Batch size: 64
- Epochs: 150

**Application**:
1. RF Analysis:
   - Feature extraction
   - Pattern recognition
   - Anomaly detection

2. Cross-sensor Correlation:
   - Signature matching
   - Feature alignment
   - Mismatch detection

#### Variational Autoencoder for Physical Analysis
**Architecture**:
- Input: Physical parameters vector
- Encoder: 3 Dense layers
- Latent space: 32 dimensions (μ, σ)
- Decoder: 3 Dense layers

**Training**:
- Loss: MSE + KL divergence
- Optimizer: Adam (lr=0.0005)
- Batch size: 16
- Epochs: 200

**Application**:
1. AMR Analysis:
   - Distribution modeling
   - Change detection
   - Family comparison

2. Physical Changes:
   - State estimation
   - Anomaly detection
   - Confidence scoring

### 3. Indicator Correlations

#### Primary Correlation Groups

1. **Maneuver-Signature Group**
   - Maneuvers detected → RF detected (80% correlation)
   - Maneuvers in gaps → RF pattern change (75% correlation)
   - POL violations → Signature changes (85% correlation)

2. **Physical-Behavioral Group**
   - Sub-satellite deployment → AMR changes (95% correlation)
   - Physical changes → Signature mismatch (90% correlation)
   - AMR changes → Maneuver detection (70% correlation)

3. **Launch-Environmental Group**
   - Objects tracked → Orbit occupancy (100% correlation)
   - Launch site → UCT correlation (60% correlation)
   - Debris analysis → AMR family (80% correlation)

#### Cross-Category Dependencies

1. **Stability Dependencies**
   ```mermaid
   graph TD
   A[Stability Change] --> B[Maneuver Detection]
   B --> C[Signature Change]
   C --> D[Physical Change]
   ```

2. **Maneuver Chain**
   ```mermaid
   graph TD
   A[Maneuver] --> B[RF Emission]
   B --> C[POL Analysis]
   C --> D[Remote Sensing]
   ```

3. **Physical Evolution**
   ```mermaid
   graph TD
   A[Sub-satellite] --> B[AMR Change]
   B --> C[Signature Mismatch]
   C --> D[Family Analysis]
   ```

#### Confidence Scoring

| Primary Indicator | Supporting Indicators | Confidence Boost |
|------------------|----------------------|------------------|
| Maneuver | RF + Signature | +15% |
| Sub-satellite | AMR + Signature | +20% |
| POL Violation | Maneuver + RF | +25% |
| Signature Change | Physical + Stability | +10% |
| AMR Change | Signature + Maneuver | +15% |

#### Alert Generation Rules

1. **High Priority**:
   - Multiple correlated indicators
   - ML confidence >95%
   - Cross-category confirmation

2. **Medium Priority**:
   - Single strong indicator
   - ML confidence >85%
   - Supporting evidence

3. **Low Priority**:
   - Weak correlations
   - ML confidence >70%
   - Isolated indicators

## Implementation Notes

1. **Data Flow**:
   ```
   Sensors → Raw Data → Feature Extraction → ML Models → Evaluators → Correlation → Alerts
   ```

2. **Update Frequency**:
   - Real-time: Maneuver, RF, UCT
   - Near-real-time: Signature, Physical
   - Batch: Launch, Environmental

3. **Performance Metrics**:
   - False Positive Rate: <1%
   - Detection Rate: >95%
   - Processing Latency: <1s

## ML Model Implementations

### 1. LSTM Autoencoder for Maneuver Detection
```python
class ManeuverLSTMAutoencoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, sequence_length=60):
        super(ManeuverLSTMAutoencoder, self).__init__()
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Dense layers for feature processing
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Thresholds
        self.reconstruction_threshold = 0.1
        self.maneuver_confidence_threshold = 0.95

    def detect_maneuver(self, trajectory_sequence):
        reconstructed = self.forward(trajectory_sequence)
        error = torch.mean((trajectory_sequence - reconstructed) ** 2)
        confidence = 1.0 - torch.sigmoid(error)
        return confidence > self.maneuver_confidence_threshold, confidence.item()
```

### 2. CNN Autoencoder for Signature Analysis
```python
class SignatureCNNAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(SignatureCNNAutoencoder, self).__init__()
        
        # Encoder CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder CNN
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # Thresholds
        self.anomaly_threshold = 0.1
        self.signature_confidence_threshold = 0.9

    def analyze_signature(self, signature_data):
        reconstructed, mu, log_var = self.forward(signature_data)
        reconstruction_loss = F.mse_loss(reconstructed, signature_data)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + kl_loss
        confidence = 1.0 - torch.sigmoid(torch.tensor(total_loss.item())).item()
        return total_loss.item() > self.anomaly_threshold, confidence
```

### 3. Variational Autoencoder for Physical Analysis
```python
class PhysicalVAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, latent_dim=32):
        super(PhysicalVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Thresholds
        self.amr_threshold = 0.15  # 15% deviation
        self.confidence_threshold = 0.95
        self.change_threshold = 0.2  # 20% change

    def analyze_amr(self, amr_data, historical_amr=None):
        current_mu, current_log_var = self.encode(amr_data)
        reconstructed = self.decode(self.reparameterize(current_mu, current_log_var))
        reconstruction_error = F.mse_loss(reconstructed, amr_data)
        
        if historical_amr is not None:
            historical_mu, _ = self.encode(historical_amr)
            change_magnitude = torch.norm(current_mu - historical_mu) / torch.norm(historical_mu)
        else:
            change_magnitude = 0.0
            
        confidence = 1.0 - torch.sigmoid(reconstruction_error).item()
        return confidence < self.confidence_threshold or change_magnitude > self.amr_threshold, confidence
```

## Indicator-Model Mapping

### 1. Maneuver Detection (LSTM)
- **Primary Indicators**:
  * Maneuvers detected
  * POL violations
  * Sensor gap maneuvers
- **Key Methods**:
  * `detect_maneuver()`: 95% confidence threshold
  * `analyze_pol()`: Historical comparison
  * `predict_gaps()`: Trajectory prediction

### 2. Signature Analysis (CNN)
- **Primary Indicators**:
  * RF emissions
  * Signature mismatches
  * Cross-sensor correlation
- **Key Methods**:
  * `analyze_signature()`: 90% confidence threshold
  * `cross_sensor_correlation()`: Optical/radar matching
  * `analyze_rf_pattern()`: Temporal analysis

### 3. Physical Analysis (VAE)
- **Primary Indicators**:
  * AMR changes
  * Sub-satellite deployment
  * Physical characteristic changes
- **Key Methods**:
  * `analyze_amr()`: 15% deviation threshold
  * `detect_subsatellite()`: 95% confidence threshold
  * `track_physical_changes()`: 20% change threshold

## Model Performance Metrics

### LSTM Autoencoder
```python
def evaluate_lstm(model, test_data):
    metrics = {
        'maneuver_detection_accuracy': 0.0,
        'pol_violation_precision': 0.0,
        'gap_prediction_rmse': 0.0
    }
    
    for sequence in test_data:
        # Maneuver detection
        detected, confidence = model.detect_maneuver(sequence)
        metrics['maneuver_detection_accuracy'] += confidence
        
        # POL analysis
        violation, pol_conf = model.analyze_pol(sequence)
        metrics['pol_violation_precision'] += pol_conf
        
        # Gap prediction
        pred, pred_conf = model.predict_gaps(sequence)
        metrics['gap_prediction_rmse'] += torch.sqrt(F.mse_loss(pred, sequence))
    
    # Average metrics
    for key in metrics:
        metrics[key] /= len(test_data)
    
    return metrics
```

### CNN Autoencoder
```python
def evaluate_cnn(model, test_data):
    metrics = {
        'signature_detection_accuracy': 0.0,
        'cross_sensor_correlation': 0.0,
        'rf_pattern_precision': 0.0
    }
    
    for data in test_data:
        # Signature analysis
        anomaly, conf = model.analyze_signature(data)
        metrics['signature_detection_accuracy'] += conf
        
        # Cross-sensor correlation
        mismatch, corr = model.cross_sensor_correlation(data['optical'], data['radar'])
        metrics['cross_sensor_correlation'] += corr
        
        # RF pattern analysis
        pattern, rf_conf = model.analyze_rf_pattern(data['rf'])
        metrics['rf_pattern_precision'] += rf_conf
    
    # Average metrics
    for key in metrics:
        metrics[key] /= len(test_data)
    
    return metrics
```

### VAE
```python
def evaluate_vae(model, test_data):
    metrics = {
        'amr_detection_accuracy': 0.0,
        'subsatellite_precision': 0.0,
        'change_detection_recall': 0.0
    }
    
    for data in test_data:
        # AMR analysis
        anomaly, conf, change = model.analyze_amr(data['current'], data['historical'])
        metrics['amr_detection_accuracy'] += conf
        
        # Subsatellite detection
        deployed, sub_conf = model.detect_subsatellite(data['primary'], data['secondary'])
        metrics['subsatellite_precision'] += sub_conf
        
        # Physical changes
        changes, points, change_conf = model.track_physical_changes(data['sequence'])
        metrics['change_detection_recall'] += change_conf
    
    # Average metrics
    for key in metrics:
        metrics[key] /= len(test_data)
    
    return metrics
```

## Model Training Configuration

### LSTM Training
```python
lstm_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'sequence_length': 60,
    'hidden_size': 128,
    'num_layers': 3
}
```

### CNN Training
```python
cnn_config = {
    'learning_rate': 0.0005,
    'batch_size': 64,
    'epochs': 150,
    'input_channels': 3,
    'latent_dim': 128
}
```

### VAE Training
```python
vae_config = {
    'learning_rate': 0.0005,
    'batch_size': 16,
    'epochs': 200,
    'input_dim': 10,
    'hidden_dim': 64,
    'latent_dim': 32
}
