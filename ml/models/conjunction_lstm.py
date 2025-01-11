import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List

class ConjunctionLSTM(nn.Module):
    """LSTM model for conjunction analysis and collision prediction"""
    
    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers for different predictions
        self.range_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1)  # Range prediction
        )
        
        self.probability_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1),  # Collision probability
            nn.Sigmoid()
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 4),  # 4 severity levels
            nn.Softmax(dim=-1)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 3),  # Uncertainties for range, prob, severity
            nn.Softplus()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Generate predictions
        range_pred = self.range_predictor(context)
        prob_pred = self.probability_predictor(context)
        severity_pred = self.severity_classifier(context)
        uncertainties = self.uncertainty_estimator(context)
        
        return {
            'range': range_pred,
            'probability': prob_pred,
            'severity': severity_pred,
            'uncertainties': uncertainties,
            'attention_weights': attention_weights
        }
    
    def predict_sequence(
        self,
        initial_sequence: torch.Tensor,
        prediction_steps: int = 10
    ) -> Dict[str, np.ndarray]:
        """Generate sequence of predictions"""
        self.eval()
        with torch.no_grad():
            # Initialize sequence
            sequence = initial_sequence.clone()
            predictions = []
            
            # Generate predictions
            for _ in range(prediction_steps):
                # Get prediction for current sequence
                pred = self.forward(sequence)
                predictions.append({
                    'range': pred['range'].numpy(),
                    'probability': pred['probability'].numpy(),
                    'severity': pred['severity'].numpy(),
                    'uncertainties': pred['uncertainties'].numpy()
                })
                
                # Update sequence with prediction
                new_step = torch.cat([
                    pred['range'],
                    pred['probability'],
                    pred['severity'],
                    sequence[:, -1, 3:]  # Keep other features unchanged
                ], dim=-1)
                sequence = torch.cat([sequence[:, 1:], new_step.unsqueeze(1)], dim=1)
            
            return predictions
    
    def calculate_risk_metrics(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate risk metrics from predictions"""
        # Extract predictions
        ranges = predictions['range']
        probabilities = predictions['probability']
        severities = predictions['severity']
        uncertainties = predictions['uncertainties']
        
        # Calculate risk metrics
        risk_score = np.mean(probabilities * np.argmax(severities, axis=-1))
        uncertainty_score = np.mean(uncertainties)
        
        # Time to closest approach
        min_range_idx = np.argmin(ranges)
        time_to_closest = min_range_idx * 60  # Assuming 60s timesteps
        
        return {
            'risk_score': float(risk_score),
            'uncertainty_score': float(uncertainty_score),
            'time_to_closest': time_to_closest,
            'min_range': float(ranges[min_range_idx])
        }

def train_conjunction_model(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    model_params: Dict = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[ConjunctionLSTM, Dict[str, List[float]]]:
    """Train the conjunction analysis model"""
    # Initialize model
    if model_params is None:
        model_params = {
            'input_size': train_data.shape[-1],
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2
        }
    
    model = ConjunctionLSTM(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        
        # Training
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Forward pass
            predictions = model(batch)
            
            # Calculate losses
            range_loss = mse_loss(predictions['range'], batch[:, -1, 0:1])
            prob_loss = mse_loss(predictions['probability'], batch[:, -1, 1:2])
            severity_loss = ce_loss(predictions['severity'], batch[:, -1, 2:6])
            
            # Combined loss
            loss = range_loss + prob_loss + severity_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate accuracy
            severity_accuracy = (
                predictions['severity'].argmax(dim=-1) == 
                batch[:, -1, 2:6].argmax(dim=-1)
            ).float().mean()
            train_accuracies.append(severity_accuracy.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                predictions = model(batch)
                
                # Calculate losses
                range_loss = mse_loss(predictions['range'], batch[:, -1, 0:1])
                prob_loss = mse_loss(predictions['probability'], batch[:, -1, 1:2])
                severity_loss = ce_loss(predictions['severity'], batch[:, -1, 2:6])
                
                loss = range_loss + prob_loss + severity_loss
                val_losses.append(loss.item())
                
                # Calculate accuracy
                severity_accuracy = (
                    predictions['severity'].argmax(dim=-1) == 
                    batch[:, -1, 2:6].argmax(dim=-1)
                ).float().mean()
                val_accuracies.append(severity_accuracy.item())
        
        # Update history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_accuracy'].append(np.mean(train_accuracies))
        history['val_accuracy'].append(np.mean(val_accuracies))
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"Train Accuracy: {history['train_accuracy'][-1]:.4f}")
            print(f"Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return model, history 