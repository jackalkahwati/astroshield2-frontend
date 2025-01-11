import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from ml.data_generation.launch_data_gen import LaunchDataGenerator

class LaunchEvaluator(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, sequence_length=24):
        """
        Neural network for launch event evaluation
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            sequence_length: Length of input sequences
        """
        super(LaunchEvaluator, self).__init__()
        
        self.sequence_length = sequence_length
        
        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Normality score branch
        self.normality_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)  # Single normality score
        )
        
        # Object count branch
        self.object_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)  # Single object count
        )
        
        # Anomaly detection branch
        self.anomaly_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 10)  # Multiple anomaly probabilities
        )
        
        # Threat assessment branch
        self.threat_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 10)  # Multiple threat features
        )

    def forward(self, x):
        """Forward pass"""
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Branch predictions
        normality_score = torch.sigmoid(self.normality_branch(context))
        object_count = torch.relu(self.object_branch(context))
        anomaly_logits = torch.sigmoid(self.anomaly_branch(context))
        threat_features = torch.sigmoid(self.threat_branch(context))
        
        return normality_score, object_count, anomaly_logits, threat_features

    def export_to_onnx(self, save_path: str, input_shape: tuple):
        """Export model to ONNX format"""
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['normality', 'objects', 'anomalies', 'threats'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'normality': {0: 'batch_size'},
                'objects': {0: 'batch_size'},
                'anomalies': {0: 'batch_size'},
                'threats': {0: 'batch_size'}
            }
        )

def train_launch_model(
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_samples: int = 10000,
    val_samples: int = 2000,
    sequence_length: int = 24
) -> LaunchEvaluator:
    """Train launch model with corrected trajectory physics"""
    
    # Initialize data generator with corrected physics
    data_gen = LaunchDataGenerator()
    
    # Generate training data
    print("Generating training data...")
    X_train, y_train = data_gen.generate_training_data(
        num_samples=train_samples,
        sequence_length=sequence_length
    )
    
    # Generate validation data
    print("Generating validation data...")
    X_val, y_val = data_gen.generate_training_data(
        num_samples=val_samples,
        sequence_length=sequence_length
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LaunchEvaluator(
        input_size=128,
        hidden_size=256,
        sequence_length=sequence_length
    )
    
    # Loss functions with weights for normal launches
    criterion_normality = nn.BCELoss(reduction='none')
    criterion_objects = nn.MSELoss(reduction='mean')
    criterion_anomaly = nn.BCEWithLogitsLoss(reduction='mean')
    criterion_threat = nn.MSELoss(reduction='mean')
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            normality_pred, object_pred, anomaly_pred, threat_pred = model(batch_X)
            
            # Calculate losses with higher weight for normal launches
            is_normal = batch_y[:, 0] > 0.8  # Success probability threshold
            normality_weights = torch.where(is_normal, torch.tensor(2.0), torch.tensor(1.0))
            normality_loss = (criterion_normality(normality_pred, batch_y[:, 0]) * normality_weights).mean()
            
            object_loss = criterion_objects(object_pred, batch_y[:, 1].unsqueeze(1))
            anomaly_loss = criterion_anomaly(anomaly_pred, batch_y[:, 2:12])
            threat_loss = criterion_threat(threat_pred, batch_y[:, 12:22])
            
            # Combined loss with emphasis on normality and anomaly detection
            loss = (0.4 * normality_loss + 
                   0.1 * object_loss + 
                   0.3 * anomaly_loss + 
                   0.2 * threat_loss)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        success_errors = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                normality_pred, object_pred, anomaly_pred, threat_pred = model(batch_X)
                
                # Calculate validation losses
                is_normal = batch_y[:, 0] > 0.8
                normality_weights = torch.where(is_normal, torch.tensor(2.0), torch.tensor(1.0))
                normality_loss = (criterion_normality(normality_pred, batch_y[:, 0]) * normality_weights).mean()
                
                object_loss = criterion_objects(object_pred, batch_y[:, 1].unsqueeze(1))
                anomaly_loss = criterion_anomaly(anomaly_pred, batch_y[:, 2:12])
                threat_loss = criterion_threat(threat_pred, batch_y[:, 12:22])
                
                loss = (0.4 * normality_loss + 
                       0.1 * object_loss + 
                       0.3 * anomaly_loss + 
                       0.2 * threat_loss)
                val_loss += loss.item()
                
                # Track success prediction errors
                success_errors.extend(
                    abs(normality_pred.squeeze().numpy() - batch_y[:, 0].numpy())
                )
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Mean Success Error: {np.mean(success_errors):.6f}")
        print("-" * 50)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/launch_evaluator.pth')
            
            # Export to ONNX
            model.export_to_onnx(
                'models/launch_evaluator.onnx',
                input_shape=(1, sequence_length, 128)
            )
            print("Saved new best model")
    
    return model

if __name__ == '__main__':
    # Train model
    print("Training launch model with corrected trajectory physics...")
    model = train_launch_model()
    print("Training complete!")
    
    # Test model
    print("\nTesting model...")
    data_gen = LaunchDataGenerator()
    
    # Generate test cases
    print("Generating test cases...")
    
    # Normal launch to LEO
    X_normal, _ = data_gen.generate_training_data(num_samples=1)
    
    # High-energy launch to GTO
    X_gto = data_gen._generate_nominal_trajectory(
        vehicle=LaunchVehicle('heavy'),
        target={'altitude': 35786000, 'inclination': np.radians(28)},
        sequence_length=24
    )[0].reshape(1, 24, 128)
    
    # Failed launch
    X_failed = data_gen._generate_nominal_trajectory(
        vehicle=LaunchVehicle('small'),
        target={'altitude': 50000, 'inclination': np.radians(28)},
        sequence_length=24
    )[0].reshape(1, 24, 128)
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        # Test normal launch
        normality_normal, _, anomaly_normal, _ = model(torch.FloatTensor(X_normal))
        print("\nNormal Launch Analysis:")
        print(f"Success Probability: {normality_normal.item():.3f}")
        print(f"Anomaly Score: {torch.mean(anomaly_normal).item():.3f}")
        
        # Test GTO launch
        normality_gto, _, anomaly_gto, _ = model(torch.FloatTensor(X_gto))
        print("\nGTO Launch Analysis:")
        print(f"Success Probability: {normality_gto.item():.3f}")
        print(f"Anomaly Score: {torch.mean(anomaly_gto).item():.3f}")
        
        # Test failed launch
        normality_failed, _, anomaly_failed, _ = model(torch.FloatTensor(X_failed))
        print("\nFailed Launch Analysis:")
        print(f"Success Probability: {normality_failed.item():.3f}")
        print(f"Anomaly Score: {torch.mean(anomaly_failed).item():.3f}")
