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

from ml.data_generation.environmental_data_gen import EnvironmentalDataGenerator

class EnvironmentalEvaluator(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, sequence_length=48):
        """
        Neural network for environmental condition evaluation
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            sequence_length: Length of input sequences
        """
        super(EnvironmentalEvaluator, self).__init__()
        
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
        
        # Eclipse prediction branch
        self.eclipse_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 3)  # [is_eclipse, umbra_prob, penumbra_prob]
        )
        
        # Occupancy prediction branch
        self.occupancy_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 4)  # [density, congestion, collision_risk, maneuver_space]
        )
        
        # Radiation prediction branch
        self.radiation_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 5)  # [total_dose, particle_flux, SAA_proximity, belt_region, solar_activity]
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
        eclipse_pred = torch.sigmoid(self.eclipse_branch(context))
        occupancy_pred = torch.sigmoid(self.occupancy_branch(context))
        radiation_pred = torch.sigmoid(self.radiation_branch(context))
        
        return eclipse_pred, occupancy_pred, radiation_pred, attention_weights

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
            output_names=['eclipse', 'occupancy', 'radiation', 'attention'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'eclipse': {0: 'batch_size'},
                'occupancy': {0: 'batch_size'},
                'radiation': {0: 'batch_size'},
                'attention': {0: 'batch_size'}
            }
        )

def train_environmental_model(
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_samples: int = 10000,
    val_samples: int = 2000,
    sequence_length: int = 48
) -> EnvironmentalEvaluator:
    """Train environmental model with corrected Van Allen belt physics"""
    
    # Initialize data generator with corrected physics
    data_gen = EnvironmentalDataGenerator()
    
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
    model = EnvironmentalEvaluator(
        input_size=128,
        hidden_size=256,
        sequence_length=sequence_length
    )
    
    # Loss functions
    criterion_eclipse = nn.BCELoss(reduction='mean')
    criterion_occupancy = nn.MSELoss(reduction='mean')
    criterion_radiation = nn.MSELoss(reduction='mean')
    
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
            eclipse_pred, occupancy_pred, radiation_pred, _ = model(batch_X)
            
            # Calculate losses with higher weight on radiation prediction
            eclipse_loss = criterion_eclipse(eclipse_pred, batch_y[:, :3])
            occupancy_loss = criterion_occupancy(occupancy_pred, batch_y[:, 3:7])
            radiation_loss = criterion_radiation(radiation_pred, batch_y[:, 7:])
            
            # Combined loss with emphasis on radiation modeling
            loss = 0.2 * eclipse_loss + 0.3 * occupancy_loss + 0.5 * radiation_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        radiation_errors = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                eclipse_pred, occupancy_pred, radiation_pred, _ = model(batch_X)
                
                # Calculate validation losses
                eclipse_loss = criterion_eclipse(eclipse_pred, batch_y[:, :3])
                occupancy_loss = criterion_occupancy(occupancy_pred, batch_y[:, 3:7])
                radiation_loss = criterion_radiation(radiation_pred, batch_y[:, 7:])
                
                loss = 0.2 * eclipse_loss + 0.3 * occupancy_loss + 0.5 * radiation_loss
                val_loss += loss.item()
                
                # Track radiation prediction errors
                radiation_errors.extend(
                    abs(radiation_pred[:, 0].numpy() - batch_y[:, 7].numpy())
                )
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Mean Radiation Error: {np.mean(radiation_errors):.6f}")
        print("-" * 50)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/environmental_evaluator.pth')
            
            # Export to ONNX
            model.export_to_onnx(
                'models/environmental_evaluator.onnx',
                input_shape=(1, sequence_length, 128)
            )
            print("Saved new best model")
    
    return model

if __name__ == '__main__':
    # Train model
    print("Training environmental model with corrected Van Allen belt physics...")
    model = train_environmental_model()
    print("Training complete!")
    
    # Test model
    print("\nTesting model...")
    data_gen = EnvironmentalDataGenerator()
    
    # Generate test cases
    print("Generating test cases...")
    
    # Low altitude case
    X_low, _ = data_gen.generate_training_data(num_samples=1)
    
    # Inner belt case
    X_inner = data_gen._create_environmental_data(
        altitude=data_gen.R * (data_gen.inner_belt['peak_L'] - 1),
        timesteps=48
    ).reshape(1, 48, 128)
    
    # Outer belt case
    X_outer = data_gen._create_environmental_data(
        altitude=data_gen.R * (data_gen.outer_belt['peak_L'] - 1),
        timesteps=48
    ).reshape(1, 48, 128)
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        # Test low altitude
        eclipse_low, occupancy_low, radiation_low, _ = model(torch.FloatTensor(X_low))
        print("\nLow Altitude Analysis:")
        print(f"Radiation Level: {radiation_low[0, 0].item():.3f}")
        print(f"Density Level: {occupancy_low[0, 0].item():.3f}")
        
        # Test inner belt
        eclipse_inner, occupancy_inner, radiation_inner, _ = model(torch.FloatTensor(X_inner))
        print("\nInner Belt Analysis:")
        print(f"Radiation Level: {radiation_inner[0, 0].item():.3f}")
        print(f"Belt Region: {radiation_inner[0, 3].item():.3f}")
        
        # Test outer belt
        eclipse_outer, occupancy_outer, radiation_outer, _ = model(torch.FloatTensor(X_outer))
        print("\nOuter Belt Analysis:")
        print(f"Radiation Level: {radiation_outer[0, 0].item():.3f}")
        print(f"Belt Region: {radiation_outer[0, 3].item():.3f}")
