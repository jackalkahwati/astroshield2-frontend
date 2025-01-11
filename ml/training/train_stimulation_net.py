import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Any
from pathlib import Path
import numpy as np

from ml.models.stimulation_net import StimulationNet
from ml.data_generation.stimulation_data_gen import StimulationDataGenerator
from ml.utils.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StimulationNetTrainer:
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 50):
        """Initialize the stimulation network trainer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GRU layers
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        }
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StimulationNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)
        
        # Initialize optimizer and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.response_criterion = nn.MSELoss()
        self.stability_criterion = nn.BCELoss()
        
        # Initialize data generator
        self.data_generator = StimulationDataGenerator()
        
        # Initialize metrics tracker
        self.metrics = ModelMetrics()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("checkpoints/stimulation_net")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, num_samples: int = 10000) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Generate data
        data = self.data_generator.generate_stimulation_data(num_samples)
        
        # Convert to tensors and ensure correct shapes
        sequences = torch.FloatTensor(data["sequences"])  # [batch_size, seq_len, input_dim]
        immediate_responses = torch.FloatTensor(data["immediate_responses"])  # [batch_size, 3]
        delayed_responses = torch.FloatTensor(data["delayed_responses"])      # [batch_size, 3]
        stability_scores = torch.FloatTensor(data["stability_scores"])        # [batch_size, 1]
        
        # Create dataset
        dataset = TensorDataset(sequences, immediate_responses, delayed_responses, stability_scores)
        
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
    
    def compute_loss(self,
                    outputs: Dict[str, torch.Tensor],
                    immediate_target: torch.Tensor,
                    delayed_target: torch.Tensor,
                    stability_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute model losses.
        
        Args:
            outputs: Model outputs
            immediate_target: Target immediate responses
            delayed_target: Target delayed responses
            stability_target: Target stability scores
            
        Returns:
            Dictionary of losses
        """
        immediate_loss = self.response_criterion(outputs["immediate_response"], immediate_target)
        delayed_loss = self.response_criterion(outputs["delayed_response"], delayed_target)
        stability_loss = self.stability_criterion(outputs["stability_score"], stability_target)
        
        # Total loss is weighted sum
        total_loss = immediate_loss + delayed_loss + stability_loss
        
        return {
            "immediate_loss": immediate_loss,
            "delayed_loss": delayed_loss,
            "stability_loss": stability_loss,
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
        total_losses = {"total": 0, "immediate": 0, "delayed": 0, "stability": 0}
        
        for sequences, immediate_target, delayed_target, stability_target in train_loader:
            # Move data to device
            sequences = sequences.to(self.device)
            immediate_target = immediate_target.to(self.device)
            delayed_target = delayed_target.to(self.device)
            stability_target = stability_target.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            losses = self.compute_loss(outputs, immediate_target, delayed_target, stability_target)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()
            
            # Update metrics
            total_losses["total"] += losses["total_loss"].item()
            total_losses["immediate"] += losses["immediate_loss"].item()
            total_losses["delayed"] += losses["delayed_loss"].item()
            total_losses["stability"] += losses["stability_loss"].item()
        
        # Compute average losses
        num_batches = len(train_loader)
        return {
            "loss": total_losses["total"] / num_batches,
            "immediate_loss": total_losses["immediate"] / num_batches,
            "delayed_loss": total_losses["delayed"] / num_batches,
            "stability_loss": total_losses["stability"] / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_losses = {"total": 0, "immediate": 0, "delayed": 0, "stability": 0}
        
        with torch.no_grad():
            for sequences, immediate_target, delayed_target, stability_target in val_loader:
                # Move data to device
                sequences = sequences.to(self.device)
                immediate_target = immediate_target.to(self.device)
                delayed_target = delayed_target.to(self.device)
                stability_target = stability_target.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                losses = self.compute_loss(outputs, immediate_target, delayed_target, stability_target)
                
                # Update metrics
                total_losses["total"] += losses["total_loss"].item()
                total_losses["immediate"] += losses["immediate_loss"].item()
                total_losses["delayed"] += losses["delayed_loss"].item()
                total_losses["stability"] += losses["stability_loss"].item()
        
        # Compute average losses
        num_batches = len(val_loader)
        return {
            "val_loss": total_losses["total"] / num_batches,
            "val_immediate_loss": total_losses["immediate"] / num_batches,
            "val_delayed_loss": total_losses["delayed"] / num_batches,
            "val_stability_loss": total_losses["stability"] / num_batches
        }
    
    def train(self, num_samples: int = 10000) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting StimulationNet training...")
        
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
    trainer = StimulationNetTrainer()
    results = trainer.train()
    
    # Print final metrics
    final_metrics = results["history"][-1]
    print("\nFinal Metrics:")
    print(f"Total Loss: {final_metrics['loss']:.4f}")
    print(f"Immediate Response Loss: {final_metrics['immediate_loss']:.4f}")
    print(f"Delayed Response Loss: {final_metrics['delayed_loss']:.4f}")
    print(f"Stability Loss: {final_metrics['stability_loss']:.4f}")
    print(f"\nValidation Metrics:")
    print(f"Val Total Loss: {final_metrics['val_loss']:.4f}")
    print(f"Val Immediate Loss: {final_metrics['val_immediate_loss']:.4f}")
    print(f"Val Delayed Loss: {final_metrics['val_delayed_loss']:.4f}")
    print(f"Val Stability Loss: {final_metrics['val_stability_loss']:.4f}") 