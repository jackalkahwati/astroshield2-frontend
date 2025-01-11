import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

from ml.models.consensus_net import ConsensusNet
from ml.utils.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsensusNetTrainer:
    def __init__(self,
                 num_evaluators: int = 5,
                 feature_dim: int = 128,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 50):
        """Initialize the consensus network trainer.
        
        Args:
            num_evaluators: Number of input evaluators
            feature_dim: Dimension of features from each evaluator
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.config = {
            "num_evaluators": num_evaluators,
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        }
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConsensusNet(
            num_evaluators=num_evaluators,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        ).to(self.device)
        
        # Initialize optimizer and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.confidence_criterion = nn.BCELoss()
        self.decision_criterion = nn.MSELoss()
        
        # Initialize metrics tracker
        self.metrics = ModelMetrics()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("checkpoints/consensus_net")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_synthetic_data(self, num_samples: int = 10000) -> tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Generate synthetic training data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (evaluator_features, confidence_targets, decision_targets)
        """
        # Generate random features for each evaluator
        evaluator_features = []
        for _ in range(self.config["num_evaluators"]):
            features = torch.randn(num_samples, self.config["feature_dim"])
            evaluator_features.append(features)
        
        # Generate synthetic targets
        # For simplicity, we'll make the ground truth a function of the average features
        avg_features = torch.stack(evaluator_features).mean(dim=0)
        confidence_targets = torch.sigmoid(avg_features.mean(dim=1)).unsqueeze(1)
        decision_targets = torch.tanh(avg_features.mean(dim=1)).unsqueeze(1)
        
        return evaluator_features, confidence_targets, decision_targets
        
    def prepare_data(self, num_samples: int = 10000) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Generate synthetic data
        evaluator_features, confidence_targets, decision_targets = self.generate_synthetic_data(num_samples)
        
        # Create dataset
        dataset = TensorDataset(*evaluator_features, confidence_targets, decision_targets)
        
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
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], confidence_target: torch.Tensor, decision_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute model losses.
        
        Args:
            outputs: Model outputs
            confidence_target: Target confidence scores
            decision_target: Target decisions
            
        Returns:
            Dictionary of losses
        """
        confidence_loss = self.confidence_criterion(outputs["confidence"], confidence_target)
        decision_loss = self.decision_criterion(outputs["decision"], decision_target)
        
        # Total loss is weighted sum
        total_loss = confidence_loss + decision_loss
        
        return {
            "confidence_loss": confidence_loss,
            "decision_loss": decision_loss,
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
        total_losses = {"total": 0, "confidence": 0, "decision": 0}
        
        for batch in train_loader:
            # Split batch into features and targets
            *evaluator_features, confidence_target, decision_target = [b.to(self.device) for b in batch]
            
            # Forward pass
            outputs = self.model(evaluator_features)
            losses = self.compute_loss(outputs, confidence_target, decision_target)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()
            
            # Update metrics
            total_losses["total"] += losses["total_loss"].item()
            total_losses["confidence"] += losses["confidence_loss"].item()
            total_losses["decision"] += losses["decision_loss"].item()
        
        # Compute average losses
        num_batches = len(train_loader)
        return {
            "loss": total_losses["total"] / num_batches,
            "confidence_loss": total_losses["confidence"] / num_batches,
            "decision_loss": total_losses["decision"] / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_losses = {"total": 0, "confidence": 0, "decision": 0}
        
        with torch.no_grad():
            for batch in val_loader:
                # Split batch into features and targets
                *evaluator_features, confidence_target, decision_target = [b.to(self.device) for b in batch]
                
                # Forward pass
                outputs = self.model(evaluator_features)
                losses = self.compute_loss(outputs, confidence_target, decision_target)
                
                # Update metrics
                total_losses["total"] += losses["total_loss"].item()
                total_losses["confidence"] += losses["confidence_loss"].item()
                total_losses["decision"] += losses["decision_loss"].item()
        
        # Compute average losses
        num_batches = len(val_loader)
        return {
            "val_loss": total_losses["total"] / num_batches,
            "val_confidence_loss": total_losses["confidence"] / num_batches,
            "val_decision_loss": total_losses["decision"] / num_batches
        }
    
    def train(self, num_samples: int = 10000) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting ConsensusNet training...")
        
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
    trainer = ConsensusNetTrainer()
    results = trainer.train()
    
    # Print final metrics
    final_metrics = results["history"][-1]
    print("\nFinal Metrics:")
    print(f"Total Loss: {final_metrics['loss']:.4f}")
    print(f"Confidence Loss: {final_metrics['confidence_loss']:.4f}")
    print(f"Decision Loss: {final_metrics['decision_loss']:.4f}")
    print(f"\nValidation Metrics:")
    print(f"Val Total Loss: {final_metrics['val_loss']:.4f}")
    print(f"Val Confidence Loss: {final_metrics['val_confidence_loss']:.4f}")
    print(f"Val Decision Loss: {final_metrics['val_decision_loss']:.4f}") 