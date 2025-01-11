import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from stability_lstm import StabilityLSTM
from maneuver_lstm import ManeuverLSTMAutoencoder
from signature_cnn import SignatureCNNAutoencoder
from physical_vae import PhysicalVAE

class ModelExporter:
    def __init__(self, base_path: str = 'models'):
        """
        Initialize model exporter
        Args:
            base_path: Base directory for saving models
        """
        self.base_path = base_path
        self.model_configs = self._get_model_configs()
        os.makedirs(base_path, exist_ok=True)
        
        # Training configurations
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all models"""
        return {
            'stability': {
                'class': StabilityLSTM,
                'input_shape': (1, 60, 6),  # (batch, sequence_length, features)
                'params': {
                    'input_size': 6,
                    'hidden_size': 128,
                    'num_layers': 3
                }
            },
            'maneuver': {
                'class': ManeuverLSTMAutoencoder,
                'input_shape': (1, 60, 6),
                'params': {
                    'input_size': 6,
                    'hidden_size': 128,
                    'num_layers': 3
                }
            },
            'signature': {
                'class': SignatureCNNAutoencoder,
                'input_shape': (1, 3, 64, 64),  # (batch, channels, height, width)
                'params': {
                    'input_channels': 3,
                    'latent_dim': 128
                }
            },
            'physical': {
                'class': PhysicalVAE,
                'input_shape': (1, 10),  # (batch, features)
                'params': {
                    'input_dim': 10,
                    'hidden_dim': 64,
                    'latent_dim': 32
                }
            }
        }
    
    def generate_sample_data(self, model_name: str) -> torch.Tensor:
        """
        Generate sample data for model training
        Args:
            model_name: Name of the model
        Returns:
            Sample data tensor
        """
        config = self.model_configs[model_name]
        shape = list(config['input_shape'])
        shape[0] = 1000  # Number of samples
        
        if model_name == 'signature':
            # Generate normalized image-like data for CNN
            return torch.rand(shape) * 2 - 1
        else:
            # Generate normalized feature data for other models
            return torch.randn(shape)
    
    def train_model(self, model: torch.nn.Module, data: torch.Tensor, model_name: str) -> None:
        """
        Train a model
        Args:
            model: PyTorch model
            data: Training data
            model_name: Name of the model
        """
        print(f"\nTraining {model_name} model...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                optimizer.zero_grad()
                
                # Forward pass (handle different model architectures)
                if model_name == 'stability':
                    # For stability model, we create target values
                    predictions, _ = model(batch)
                    # Create target values (assuming stable orbits in training)
                    targets = torch.ones(predictions.shape[0], 3).to(predictions.device)
                    targets[:, 0] = 0.9  # High stability score
                    targets[:, 1] = 0.1  # Low family deviation
                    targets[:, 2] = 0.1  # Low anomaly score
                    loss = criterion(predictions, targets)
                elif model_name == 'maneuver':
                    # For maneuver model (autoencoder)
                    reconstructed = model(batch)
                    loss = criterion(reconstructed, batch)
                elif model_name == 'signature':
                    output, mu, log_var = model(batch)
                    loss = criterion(output, batch)
                    # Add KL divergence loss for VAE
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss += 0.1 * kl_loss
                elif model_name == 'physical':
                    output, mu, log_var = model(batch)
                    loss = criterion(output, batch)
                    # Add KL divergence loss for VAE
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss += 0.1 * kl_loss
                
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
    
    def export_to_onnx(self, model: torch.nn.Module, model_name: str) -> None:
        """
        Export model to ONNX format with proper configurations
        Args:
            model: Trained PyTorch model
            model_name: Name of the model
        """
        print(f"\nExporting {model_name} model to ONNX...")
        
        config = self.model_configs[model_name]
        dummy_input = torch.randn(config['input_shape'])
        save_path = os.path.join(self.base_path, f"{model_name}_model.onnx")
        
        # Model-specific export configurations
        if model_name in ['stability', 'maneuver']:
            # LSTM models need special handling for dynamic sequences
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        elif model_name == 'signature':
            # CNN model with image input
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            # VAE model
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export the model with appropriate configuration
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"Model exported to: {save_path}")
        
        # Verify the exported model
        try:
            import onnx
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model check passed for {model_name}")
        except Exception as e:
            print(f"Warning: ONNX model verification failed: {str(e)}")
    
    def save_pytorch_model(self, model: torch.nn.Module, model_name: str) -> None:
        """
        Save PyTorch model
        Args:
            model: Trained PyTorch model
            model_name: Name of the model
        """
        save_path = os.path.join(self.base_path, f"{model_name}_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"PyTorch model saved to: {save_path}")
    
    def export_all_models(self) -> None:
        """Export all evaluator models"""
        print("Starting model export process...")
        
        for model_name, config in self.model_configs.items():
            print(f"\nProcessing {model_name} model...")
            
            # Initialize model
            model = config['class'](**config['params'])
            
            # Generate and prepare data
            train_data = self.generate_sample_data(model_name)
            
            # Train model
            self.train_model(model, train_data, model_name)
            
            # Save PyTorch model
            self.save_pytorch_model(model, model_name)
            
            # Export to ONNX
            self.export_to_onnx(model, model_name)
        
        print("\nAll models exported successfully!")

def main():
    """Main function to export all models"""
    try:
        exporter = ModelExporter()
        exporter.export_all_models()
    except Exception as e:
        print(f"Error during model export: {str(e)}")

if __name__ == '__main__':
    main()
