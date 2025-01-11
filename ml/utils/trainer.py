import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
import numpy as np
from tqdm import tqdm
import wandb
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import json
from collections import defaultdict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    model_name: str
    num_epochs: int
    learning_rate: float
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize the model trainer.
        
        Args:
            model: The PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: The optimizer to use
            criterion: The loss function
            config: Training configuration dictionary
            scheduler: Optional learning rate scheduler
            metrics: Optional dictionary of metric functions
            device: Optional device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(os.path.dirname(self.config["checkpoint_path"]), exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_metrics = defaultdict(float)
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update metrics
                for metric_name, metric_fn in self.metrics.items():
                    total_metrics[metric_name] += metric_fn(output, target)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    **{name: value / (batch_idx + 1) for name, value in total_metrics.items()}
                })
        
        return {
            'loss': total_loss / num_batches,
            **{name: value / num_batches for name, value in total_metrics.items()}
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_metrics = defaultdict(float)
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Update metrics
                for metric_name, metric_fn in self.metrics.items():
                    total_metrics[metric_name] += metric_fn(output, target)
        
        return {
            'loss': total_loss / num_batches,
            **{name: value / num_batches for name, value in total_metrics.items()}
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model for the specified number of epochs."""
        history = defaultdict(list)
        
        for epoch in range(1, self.config["num_epochs"] + 1):
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            metrics_str = f"epoch: {epoch:.4f}"
            for name, value in train_metrics.items():
                metrics_str += f" - train_{name}: {value:.4f}"
                history[f'train_{name}'].append(value)
            for name, value in val_metrics.items():
                metrics_str += f" - val_{name}: {value:.4f}"
                history[f'val_{name}'].append(value)
            
            logger.info(metrics_str)
            
            # Check for improvement
            val_loss = val_metrics['loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), self.config["checkpoint_path"])
                logger.info(f"Saved best model checkpoint to {self.config['checkpoint_path']}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get("early_stopping_patience", float('inf')):
                logger.info("Early stopping triggered")
                break
        
        return dict(history) 