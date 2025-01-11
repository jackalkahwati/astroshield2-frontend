import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

class ModelMetrics:
    @staticmethod
    def mse_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Mean squared error loss."""
        return F.mse_loss(predictions, targets, reduction=reduction)
    
    @staticmethod
    def mae_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Mean absolute error loss."""
        return F.l1_loss(predictions, targets, reduction=reduction)
    
    @staticmethod
    def rmse_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Root mean squared error loss."""
        return torch.sqrt(F.mse_loss(predictions, targets))
    
    @staticmethod
    def huber_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        delta: float = 1.0
    ) -> torch.Tensor:
        """Huber loss for robust regression."""
        return F.huber_loss(predictions, targets, delta=delta)
    
    @staticmethod
    def binary_cross_entropy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Binary cross entropy loss."""
        return F.binary_cross_entropy(predictions, targets, reduction=reduction)
    
    @staticmethod
    def categorical_cross_entropy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Categorical cross entropy loss."""
        return F.cross_entropy(predictions, targets, reduction=reduction)
    
    @staticmethod
    def accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Classification accuracy."""
        if predictions.shape != targets.shape:
            predictions = torch.argmax(predictions, dim=1)
        return (predictions == targets).float().mean()
    
    @staticmethod
    def precision_recall_f1(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        average: str = 'macro'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Calculate precision, recall, and F1 score."""
        if predictions.shape != targets.shape:
            predictions = torch.argmax(predictions, dim=1)
        
        return precision_recall_fscore_support(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average=average
        )
    
    @staticmethod
    def roc_auc(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        multi_class: str = 'ovr'
    ) -> float:
        """Calculate ROC AUC score."""
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class case
            return roc_auc_score(
                targets.cpu().numpy(),
                F.softmax(predictions, dim=1).cpu().numpy(),
                multi_class=multi_class
            )
        else:
            # Binary case
            return roc_auc_score(
                targets.cpu().numpy(),
                predictions.cpu().numpy()
            )
    
    @staticmethod
    def confusion_matrix(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        if predictions.shape != targets.shape:
            predictions = torch.argmax(predictions, dim=1)
        
        return confusion_matrix(
            targets.cpu().numpy(),
            predictions.cpu().numpy()
        )
    
    @staticmethod
    def r2_score(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate R² score for regression."""
        target_mean = torch.mean(targets)
        ss_tot = torch.sum((targets - target_mean) ** 2)
        ss_res = torch.sum((targets - predictions) ** 2)
        return 1 - ss_res / ss_tot
    
    @staticmethod
    def explained_variance(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate explained variance score."""
        target_var = torch.var(targets)
        explained_var = torch.var(targets - predictions)
        return 1 - explained_var / target_var
    
    @staticmethod
    def mean_absolute_percentage_error(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate mean absolute percentage error."""
        return torch.mean(torch.abs((targets - predictions) / targets)) * 100
    
    @staticmethod
    def weighted_mse_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Weighted mean squared error loss."""
        return torch.mean(weights * (predictions - targets) ** 2)
    
    @staticmethod
    def focal_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """Focal loss for imbalanced classification."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return torch.mean(focal_loss)
    
    @classmethod
    def get_all_metrics(
        cls,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_type: str = 'classification'
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Calculate all relevant metrics based on task type."""
        metrics = {}
        
        if task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = cls.accuracy(predictions, targets).item()
            precision, recall, f1, _ = cls.precision_recall_f1(predictions, targets)
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            
            try:
                metrics['roc_auc'] = cls.roc_auc(predictions, targets)
            except ValueError:
                # ROC AUC might not be applicable for some cases
                pass
            
            metrics['confusion_matrix'] = cls.confusion_matrix(predictions, targets)
            
        elif task_type == 'regression':
            # Regression metrics
            metrics['mse'] = cls.mse_loss(predictions, targets).item()
            metrics['mae'] = cls.mae_loss(predictions, targets).item()
            metrics['rmse'] = cls.rmse_loss(predictions, targets).item()
            metrics['r2'] = cls.r2_score(predictions, targets).item()
            metrics['explained_variance'] = cls.explained_variance(predictions, targets).item()
            
            # Calculate MAPE only if targets don't contain zeros
            if not torch.any(targets == 0):
                metrics['mape'] = cls.mean_absolute_percentage_error(predictions, targets).item()
        
        return metrics 