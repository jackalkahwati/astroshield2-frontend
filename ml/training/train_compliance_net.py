import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

from ml.models.compliance_net import ComplianceNet
from ml.data_generation.compliance_data_gen import ComplianceDataGenerator
from ml.utils.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceNetTrainer:
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 num_rules: int = 32,
                 num_layers: int = 4,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 50):
        """Initialize the compliance network trainer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_rules: Number of compliance rules
            num_layers: Number of transformer layers
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_rules": num_rules,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        }
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ComplianceNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_rules=num_rules,
            num_layers=num_layers
        ).to(self.device)
        
        # Initialize optimizer and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.rule_criterion = nn.BCELoss()
        self.severity_criterion = nn.MSELoss()
        
        # Initialize data generator
        self.data_generator = ComplianceDataGenerator()
        
        # Initialize metrics tracker
        self.metrics = ModelMetrics()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("checkpoints/compliance_net")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, num_samples: int = 10000) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Generate data
        data = self.data_generator.generate_compliance_data(num_samples)
        
        # Convert to tensors and ensure correct shapes
        sequences = torch.FloatTensor(data["sequences"])  # [batch_size, seq_len, input_dim]
        rules = torch.FloatTensor(data["rules"])         # [batch_size, num_rules]
        scores = torch.FloatTensor(data["scores"])       # [batch_size, 1]
        
        # Create dataset
        dataset = TensorDataset(sequences, rules, scores)
        
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
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], rule_target: torch.Tensor, severity_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute model losses.
        
        Args:
            outputs: Model outputs
            rule_target: Target rule compliance scores
            severity_target: Target severity scores
            
        Returns:
            Dictionary of losses
        """
        rule_loss = self.rule_criterion(outputs["rule_compliance"], rule_target)
        severity_loss = self.severity_criterion(outputs["violation_severity"], severity_target)
        
        # Total loss is weighted sum
        total_loss = rule_loss + severity_loss
        
        return {
            "rule_loss": rule_loss,
            "severity_loss": severity_loss,
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
        total_losses = {"total": 0, "rule": 0, "severity": 0}
        
        for sequences, rule_target, severity_target in train_loader:
            # Move data to device
            sequences = sequences.to(self.device)
            rule_target = rule_target.to(self.device)
            severity_target = severity_target.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            losses = self.compute_loss(outputs, rule_target, severity_target)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()
            
            # Update metrics
            total_losses["total"] += losses["total_loss"].item()
            total_losses["rule"] += losses["rule_loss"].item()
            total_losses["severity"] += losses["severity_loss"].item()
        
        # Compute average losses
        num_batches = len(train_loader)
        return {
            "loss": total_losses["total"] / num_batches,
            "rule_loss": total_losses["rule"] / num_batches,
            "severity_loss": total_losses["severity"] / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_losses = {"total": 0, "rule": 0, "severity": 0}
        
        with torch.no_grad():
            for sequences, rule_target, severity_target in val_loader:
                # Move data to device
                sequences = sequences.to(self.device)
                rule_target = rule_target.to(self.device)
                severity_target = severity_target.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                losses = self.compute_loss(outputs, rule_target, severity_target)
                
                # Update metrics
                total_losses["total"] += losses["total_loss"].item()
                total_losses["rule"] += losses["rule_loss"].item()
                total_losses["severity"] += losses["severity_loss"].item()
        
        # Compute average losses
        num_batches = len(val_loader)
        return {
            "val_loss": total_losses["total"] / num_batches,
            "val_rule_loss": total_losses["rule"] / num_batches,
            "val_severity_loss": total_losses["severity"] / num_batches
        }
    
    def train(self, num_samples: int = 10000) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting ComplianceNet training...")
        
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
    trainer = ComplianceNetTrainer()
    results = trainer.train()
    
    # Print final metrics
    final_metrics = results["history"][-1]
    print("\nFinal Metrics:")
    print(f"Total Loss: {final_metrics['loss']:.4f}")
    print(f"Rule Loss: {final_metrics['rule_loss']:.4f}")
    print(f"Severity Loss: {final_metrics['severity_loss']:.4f}")
    print(f"\nValidation Metrics:")
    print(f"Val Total Loss: {final_metrics['val_loss']:.4f}")
    print(f"Val Rule Loss: {final_metrics['val_rule_loss']:.4f}")
    print(f"Val Severity Loss: {final_metrics['val_severity_loss']:.4f}") 