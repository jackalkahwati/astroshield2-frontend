import torch
import numpy as np
from stability_lstm import StabilityLSTM
import os

def train_stability_model(model: StabilityLSTM, train_data: np.ndarray, epochs: int = 100):
    """
    Train the stability model
    Args:
        model: StabilityLSTM model instance
        train_data: Training data of shape (num_samples, sequence_length, input_size)
        epochs: Number of training epochs
    """
    # Convert data to tensor
    train_tensor = torch.FloatTensor(train_data)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = model(train_tensor)
        
        # Compute loss (example: predict stability close to 1.0)
        target = torch.ones(len(train_tensor), 3)  # Example target
        loss = criterion(predictions, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def export_model(model_path: str = 'models'):
    """
    Train and export the stability model to ONNX format
    Args:
        model_path: Directory to save the model
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize model
    model = StabilityLSTM()
    
    # Generate example training data
    # In practice, this would be your real orbital data
    sequence_length = 60
    input_size = 6  # [x, y, z, vx, vy, vz]
    num_samples = 1000
    
    # Example: Generate random orbital data
    train_data = np.random.randn(num_samples, sequence_length, input_size)
    
    # Train the model
    train_stability_model(model, train_data)
    
    # Save PyTorch model
    torch.save(model.state_dict(), os.path.join(model_path, 'stability_model.pth'))
    
    # Export to ONNX
    input_shape = (1, sequence_length, input_size)  # (batch_size, seq_len, features)
    model.export_to_onnx(
        os.path.join(model_path, 'stability_model.onnx'),
        input_shape
    )
    
    print(f"Model exported successfully to {model_path}")
    print("Files created:")
    print(f" - {os.path.join(model_path, 'stability_model.pth')} (PyTorch model)")
    print(f" - {os.path.join(model_path, 'stability_model.onnx')} (ONNX model)")

if __name__ == '__main__':
    export_model()
