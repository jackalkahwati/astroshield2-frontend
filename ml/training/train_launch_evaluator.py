import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Any
from pathlib import Path

from ml.models.launch_evaluator import LaunchEvaluator
from ml.data_generation.launch_data_gen import LaunchDataGenerator
from ml.utils.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaunchEvaluatorTrainer:
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 4,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 50):
        """Initialize the launch evaluator trainer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of launch pattern classes
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        }
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LaunchEvaluator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize data generator
        self.data_generator = LaunchDataGenerator()
        
        # Initialize metrics tracker
        self.metrics = ModelMetrics()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("checkpoints/launch_evaluator")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, num_samples: int = 10000) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Generate data
        data = self.data_generator.generate_launch_data(num_samples)
        
        # Convert to tensors and ensure correct shapes
        sequences = torch.FloatTensor(data["sequences"])  # [batch_size, seq_len, input_dim]
        labels = torch.LongTensor(data["labels"]).squeeze()  # [batch_size]
        
        # Create dataset
        dataset = TensorDataset(sequences, labels)
        
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
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs["logits"], labels)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs["logits"].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "loss": total_loss / len(train_loader),
            "accuracy": 100. * correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs["logits"], labels)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs["logits"].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            "val_loss": total_loss / len(val_loader),
            "val_accuracy": 100. * correct / total
        }
    
    def train(self, num_samples: int = 10000) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting LaunchEvaluator training...")
        
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
                       f"Accuracy: {metrics['accuracy']:.2f}% - "
                       f"Val Loss: {metrics['val_loss']:.4f} - "
                       f"Val Accuracy: {metrics['val_accuracy']:.2f}%")
            
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
    trainer = LaunchEvaluatorTrainer()
    results = trainer.train()
    
    # Print final metrics
    final_metrics = results["history"][-1]
    print("\nFinal Metrics:")
    print(f"Loss: {final_metrics['loss']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Val Loss: {final_metrics['val_loss']:.4f}")
    print(f"Val Accuracy: {final_metrics['val_accuracy']:.2f}%") 