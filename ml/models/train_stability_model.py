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

from ml.data_generation.stability_data_gen import StabilityDataGenerator
from ml.models.stability_lstm import StabilityLSTM

def train_stability_model(
    num_epochs: int = 50,  # Reduced for faster iteration
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_samples: int = 5000,  # Reduced for faster iteration
    val_samples: int = 1000,   # Reduced for faster iteration
    sequence_length: int = 60
) -> StabilityLSTM:
    """Train stability model with corrected physics"""
    
    print("\nInitializing Stability Model Training...")
    print(f"Configuration:")
    print(f"- Epochs: {num_epochs}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Training Samples: {train_samples}")
    print(f"- Validation Samples: {val_samples}")
    print(f"- Sequence Length: {sequence_length}")
    
    # Initialize data generator with corrected physics
    print("\nInitializing data generator...")
    data_gen = StabilityDataGenerator()
    
    # Generate training data
    print("\nGenerating training data...")
    X_train, y_train = data_gen.generate_training_data(
        num_samples=train_samples,
        sequence_length=sequence_length
    )
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Generate validation data
    print("\nGenerating validation data...")
    X_val, y_val = data_gen.generate_training_data(
        num_samples=val_samples,
        sequence_length=sequence_length
    )
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    
    # Convert to PyTorch tensors
    print("\nConverting to PyTorch tensors...")
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    print("\nInitializing model...")
    model = StabilityLSTM(
        input_size=6,
        hidden_size=128,
        num_layers=3,
        sequence_length=sequence_length
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training loop...")
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions, _ = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        stability_errors = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions, _ = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
                # Track stability prediction errors
                stability_errors.extend(
                    abs(predictions[:, 0].numpy() - batch_y[:, 0].numpy())
                )
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Mean Stability Error: {np.mean(stability_errors):.6f}")
        print(f"Max Stability Error: {np.max(stability_errors):.6f}")
        print(f"Min Stability Error: {np.min(stability_errors):.6f}")
        print("-" * 50)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("\nSaving new best model...")
            
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            
            # Save PyTorch model
            torch.save(model.state_dict(), 'models/stability_model.pth')
            print("Saved PyTorch model")
            
            # Export to ONNX
            try:
                model.export_to_onnx(
                    'models/stability_model.onnx',
                    input_shape=(1, sequence_length, 6)
                )
                print("Exported to ONNX format")
            except Exception as e:
                print(f"Error exporting to ONNX: {str(e)}")
    
    return model

if __name__ == '__main__':
    # Train model
    print("Starting stability model training with corrected physics...")
    model = train_stability_model()
    print("\nTraining complete!")
    
    # Test model
    print("\nTesting model...")
    data_gen = StabilityDataGenerator()
    
    # Generate test cases
    print("\nGenerating test cases...")
    
    # Circular orbit
    print("\nTesting circular orbit...")
    circular_orbit, _ = data_gen.generate_training_data(num_samples=1)
    
    # Elliptical orbit
    print("\nTesting elliptical orbit...")
    data_gen.max_eccentricity = 0.3  # Temporarily increase eccentricity
    elliptical_orbit, _ = data_gen.generate_training_data(num_samples=1)
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        # Test circular orbit
        circular_pred = model.analyze_stability(torch.FloatTensor(circular_orbit[0]))
        print("\nCircular Orbit Analysis:")
        print(f"Stability Score: {circular_pred['stability_score']:.3f}")
        print(f"Family Deviation: {circular_pred['family_deviation']:.3f}")
        print(f"Anomaly Score: {circular_pred['anomaly_score']:.3f}")
        print(f"Confidence: {circular_pred['confidence']:.3f}")
        
        # Test elliptical orbit
        elliptical_pred = model.analyze_stability(torch.FloatTensor(elliptical_orbit[0]))
        print("\nElliptical Orbit Analysis:")
        print(f"Stability Score: {elliptical_pred['stability_score']:.3f}")
        print(f"Family Deviation: {elliptical_pred['family_deviation']:.3f}")
        print(f"Anomaly Score: {elliptical_pred['anomaly_score']:.3f}")
        print(f"Confidence: {elliptical_pred['confidence']:.3f}")
        
        # Verify physics
        if circular_pred['stability_score'] > elliptical_pred['stability_score']:
            print("\nPhysics validation PASSED: Circular orbit correctly identified as more stable")
        else:
            print("\nPhysics validation FAILED: Circular orbit not identified as more stable")
