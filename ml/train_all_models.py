import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple

from ml.models.track_evaluator import TrackEvaluator
from ml.models.stability_evaluator import StabilityEvaluator 
from ml.models.maneuver_planner import ManeuverPlanner
from ml.models.physical_properties_net import PhysicalPropertiesNet
from ml.models.environmental_evaluator import EnvironmentalEvaluator
from ml.models.launch_evaluator import LaunchEvaluator
from ml.models.consensus_net import ConsensusNet

from ml.data_generation.track_data_gen import TrackDataGenerator
from ml.data_generation.stability_data_gen import StabilityDataGenerator
from ml.data_generation.maneuver_data_gen import ManeuverDataGenerator
from ml.data_generation.physical_data_gen import PhysicalPropertiesGenerator
from ml.data_generation.environmental_data_gen import EnvironmentalDataGenerator
from ml.data_generation.launch_data_gen import LaunchDataGenerator

class UnifiedTrainer:
    def __init__(self):
        # Training parameters
        self.batch_size = 32
        self.max_epochs = 50
        self.patience = 5
        self.learning_rate = 1e-4
        
        # Initialize models
        self.track_evaluator = TrackEvaluator(input_dim=64)
        self.stability_evaluator = StabilityEvaluator()
        self.maneuver_planner = ManeuverPlanner()
        self.physical_properties_net = PhysicalPropertiesNet()
        self.environmental_evaluator = EnvironmentalEvaluator()
        self.launch_evaluator = LaunchEvaluator()
        
        # Initialize data generators
        self.track_generator = TrackDataGenerator()
        self.stability_generator = StabilityDataGenerator()
        self.maneuver_generator = ManeuverDataGenerator()
        self.physical_generator = PhysicalPropertiesGenerator()
        self.environmental_generator = EnvironmentalDataGenerator()
        self.launch_generator = LaunchDataGenerator()
        
        # Initialize consensus model
        self.consensus_net = ConsensusNet(
            input_dims={
                'track': 64,
                'stability': 32,
                'maneuver': 48,
                'physical': 128,
                'environmental': 96,
                'launch': 64
            }
        )
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._move_models_to_device()
        
    def _move_models_to_device(self):
        """Move all models to the configured device."""
        self.track_evaluator.to(self.device)
        self.stability_evaluator.to(self.device)
        self.maneuver_planner.to(self.device)
        self.physical_properties_net.to(self.device)
        self.environmental_evaluator.to(self.device)
        self.launch_evaluator.to(self.device)
        self.consensus_net.to(self.device)
        
    def generate_data(self, model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate training data for a specific model."""
        print(f"Generating data for {model_name}...")
        
        if model_name == 'track':
            X, y = self.track_generator.generate_training_data()
        elif model_name == 'stability':
            X, y = self.stability_generator.generate_training_data()
        elif model_name == 'maneuver':
            X, y = self.maneuver_generator.generate_training_data()
        elif model_name == 'physical':
            X, y = self.physical_generator.generate_training_data()
        elif model_name == 'environmental':
            X, y = self.environmental_generator.generate_training_data()
        elif model_name == 'launch':
            X, y = self.launch_generator.generate_training_data()
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Split into train/val
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        return (X_train, y_train), (X_val, y_val)
        
    def train_model(self, model_name: str, model: nn.Module, train_data: Tuple, val_data: Tuple) -> None:
        """Train a single model.
        
        Args:
            model_name: Name of the model being trained
            model: The model to train
            train_data: Training data tuple (features, labels)
            val_data: Validation data tuple (features, labels)
        """
        print(f"\nTraining {model_name}...")
        
        # Unpack data
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        
        # Handle different output formats
        if model_name == "stability":
            y_train = {k: torch.FloatTensor(v) for k, v in y_train.items()}
            y_val = {k: torch.FloatTensor(v) for k, v in y_val.items()}
        else:
            y_train = torch.FloatTensor(y_train)
            y_val = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = model(batch_X)
                
                # Compute loss
                if model_name == "stability":
                    loss = model.compute_loss(outputs, batch_y)
                else:
                    loss = model.compute_loss(outputs, batch_y)
                
                # Backward pass
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    if model_name == "stability":
                        loss = model.compute_loss(outputs, batch_y)
                    else:
                        loss = model.compute_loss(outputs, batch_y)
                    val_loss += loss.item()
            
            # Print progress
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model.save_model(f"models/{model_name}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
    def train_all_models(self):
        """Train all models in sequence."""
        models = {
            'track': self.track_evaluator,
            'stability': self.stability_evaluator,
            'maneuver': self.maneuver_planner,
            'physical': self.physical_properties_net,
            'environmental': self.environmental_evaluator,
            'launch': self.launch_evaluator
        }
        
        # Train each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name} model...")
            train_data, val_data = self.generate_data(model_name)
            self.train_model(model_name, model, train_data, val_data)
            
        # Train consensus model
        print("\nTraining consensus model...")
        self.train_consensus_model()
        
    def train_consensus_model(self):
        """Train the consensus model using outputs from all other models."""
        # Generate consensus training data
        print("Generating consensus training data...")
        
        # Get predictions from all models
        model_outputs = {}
        for model_name in ['track', 'stability', 'maneuver', 'physical', 'environmental', 'launch']:
            train_data, _ = self.generate_data(model_name)
            X_train, _ = train_data
            
            model = getattr(self, f"{model_name}_evaluator")
            if model_name == 'physical':
                model = self.physical_properties_net
            elif model_name == 'maneuver':
                model = self.maneuver_planner
                
            with torch.no_grad():
                model.eval()
                model_outputs[model_name] = model(X_train.to(self.device))
                
        # Combine outputs for consensus training
        consensus_input = torch.cat([
            outputs for outputs in model_outputs.values()
        ], dim=-1)
        
        # Generate consensus labels (example: weighted average of individual predictions)
        consensus_labels = torch.zeros((len(consensus_input), 32)).to(self.device)
        for i, outputs in enumerate(model_outputs.values()):
            consensus_labels += outputs * (1.0 / len(model_outputs))
            
        # Train consensus model
        consensus_dataset = TensorDataset(consensus_input, consensus_labels)
        train_size = int(0.8 * len(consensus_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            consensus_dataset, 
            [train_size, len(consensus_dataset) - train_size]
        )
        
        self.train_model(
            'consensus',
            self.consensus_net,
            (train_dataset[:][0], train_dataset[:][1]),
            (val_dataset[:][0], val_dataset[:][1])
        )

if __name__ == "__main__":
    trainer = UnifiedTrainer()
    trainer.train_all_models() 