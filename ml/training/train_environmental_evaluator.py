import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Any
from pathlib import Path

from ml.models.environmental_evaluator import EnvironmentalEvaluator
from ml.data_generation.environmental_data_gen import EnvironmentalDataGenerator
from ml.utils.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentalEvaluatorTrainer:
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 50):
        """Initialize the environmental evaluator trainer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        }
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnvironmentalEvaluator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize loss functions for each prediction type
        self.radiation_criterion = nn.MSELoss()
        self.debris_criterion = nn.MSELoss()
        self.solar_criterion = nn.MSELoss()
        self.magnetic_criterion = nn.MSELoss()
        
        # Initialize data generator
        self.data_generator = EnvironmentalDataGenerator()
        
        # Initialize metrics tracker
        self.metrics = ModelMetrics()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("checkpoints/environmental_evaluator")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, num_samples: int = 10000) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Generate data
        data = self.data_generator.generate_environmental_data(num_samples)
        
        # Convert to tensors and ensure correct shapes
        sequences = torch.FloatTensor(data["sequences"])  # [batch_size, seq_len, input_dim]
        metrics = torch.FloatTensor(data["metrics"])      # [batch_size, num_metrics]
        
        # Create dataset
        dataset = TensorDataset(sequences, metrics)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"]
        )
        
        return train_loader, val_loader
    
    def compute_losses(self, outputs: Dict[str, torch.Tensor], targets: tuple) -> Dict[str, torch.Tensor]:
        """Compute losses for each prediction type.
        
        Args:
            outputs: Model outputs
            targets: Tuple of target tensors
            
        Returns:
            Dictionary of losses
        """
        radiation_target, debris_target, solar_target, magnetic_target = targets
        
        radiation_loss = self.radiation_criterion(outputs["radiation_level"], radiation_target)
        debris_loss = self.debris_criterion(outputs["debris_density"], debris_target)
        solar_loss = self.solar_criterion(outputs["solar_activity"], solar_target)
        magnetic_loss = self.magnetic_criterion(outputs["magnetic_field"], magnetic_target)
        
        # Compute total loss as weighted sum
        total_loss = radiation_loss + debris_loss + solar_loss + magnetic_loss
        
        return {
            "radiation_loss": radiation_loss,
            "debris_loss": debris_loss,
            "solar_loss": solar_loss,
            "magnetic_loss": magnetic_loss,
            "total_loss": total_loss
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_losses = {"total": 0, "radiation": 0, "debris": 0, "solar": 0, "magnetic": 0}
        
        for sequences, metrics in train_loader:
            # Move data to device
            sequences = sequences.to(self.device)
            metrics = metrics.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            losses = self.compute_losses(outputs, metrics)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()
            
            # Update metrics
            total_losses["total"] += losses["total_loss"].item()
            total_losses["radiation"] += losses["radiation_loss"].item()
            total_losses["debris"] += losses["debris_loss"].item()
            total_losses["solar"] += losses["solar_loss"].item()
            total_losses["magnetic"] += losses["magnetic_loss"].item()
        
        # Compute average losses
        num_batches = len(train_loader)
        return {
            "loss": total_losses["total"] / num_batches,
            "radiation_loss": total_losses["radiation"] / num_batches,
            "debris_loss": total_losses["debris"] / num_batches,
            "solar_loss": total_losses["solar"] / num_batches,
            "magnetic_loss": total_losses["magnetic"] / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_losses = {"total": 0, "radiation": 0, "debris": 0, "solar": 0, "magnetic": 0}
        
        with torch.no_grad():
            for sequences, metrics in val_loader:
                # Move data to device
                sequences = sequences.to(self.device)
                metrics = metrics.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                losses = self.compute_losses(outputs, metrics)
                
                # Update metrics
                total_losses["total"] += losses["total_loss"].item()
                total_losses["radiation"] += losses["radiation_loss"].item()
                total_losses["debris"] += losses["debris_loss"].item()
                total_losses["solar"] += losses["solar_loss"].item()
                total_losses["magnetic"] += losses["magnetic_loss"].item()
        
        # Compute average losses
        num_batches = len(val_loader)
        return {
            "val_loss": total_losses["total"] / num_batches,
            "val_radiation_loss": total_losses["radiation"] / num_batches,
            "val_debris_loss": total_losses["debris"] / num_batches,
            "val_solar_loss": total_losses["solar"] / num_batches,
            "val_magnetic_loss": total_losses["magnetic"] / num_batches
        }
    
    def train(self, num_samples: int = 10000) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting EnvironmentalEvaluator training...")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(num_samples)
        
        # Training loop
        best_val_loss = float('inf')
        history = []
        
        for epoch in range(self.config["num_epochs"]):
            # Train and validate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            history.append(metrics)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                       f"Loss: {metrics['loss']:.4f} - "
                       f"Val Loss: {metrics['val_loss']:.4f}")
            
            # Save best model
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                self.save_model("best.pt")
        
        # Save final model
        self.save_model("final.pt")
        logger.info("Training completed!")
        
        return {
            "history": history,
            "config": self.config
        }
    
    def save_model(self, filename: str):
        """Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, filename: str):
        """Load model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']

if __name__ == "__main__":
    # Train model
    trainer = EnvironmentalEvaluatorTrainer()
    results = trainer.train()
    
    # Print final metrics
    final_metrics = results["history"][-1]
    print("\nFinal Metrics:")
    print(f"Total Loss: {final_metrics['loss']:.4f}")
    print(f"Radiation Loss: {final_metrics['radiation_loss']:.4f}")
    print(f"Debris Loss: {final_metrics['debris_loss']:.4f}")
    print(f"Solar Loss: {final_metrics['solar_loss']:.4f}")
    print(f"Magnetic Loss: {final_metrics['magnetic_loss']:.4f}")
    print(f"\nValidation Metrics:")
    print(f"Val Total Loss: {final_metrics['val_loss']:.4f}")
    print(f"Val Radiation Loss: {final_metrics['val_radiation_loss']:.4f}")
    print(f"Val Debris Loss: {final_metrics['val_debris_loss']:.4f}")
    print(f"Val Solar Loss: {final_metrics['val_solar_loss']:.4f}")
    print(f"Val Magnetic Loss: {final_metrics['val_magnetic_loss']:.4f}") 