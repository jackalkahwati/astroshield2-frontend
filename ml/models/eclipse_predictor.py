import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class EclipsePredictor(nn.Module):
    """Model for predicting eclipse periods and their effects"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Eclipse state predictor
        self.eclipse_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 3)  # [sunlight, penumbra, umbra]
        )
        
        # Thermal predictor
        self.thermal_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)  # Temperature in Kelvin
        )
        
        # Power predictor
        self.power_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)  # [battery_state, power_consumption]
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 4),  # Uncertainties for each prediction
            nn.Softplus()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Process sequence through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Generate predictions
        eclipse_state = self.eclipse_predictor(context)
        temperature = self.thermal_predictor(context)
        power_state = self.power_predictor(context)
        uncertainties = self.uncertainty(context)
        
        return {
            'eclipse_state': eclipse_state,
            'temperature': temperature,
            'power_state': power_state,
            'uncertainties': uncertainties,
            'attention_weights': attention_weights,
            'hidden': hidden
        }
    
    def predict_sequence(
        self,
        initial_sequence: torch.Tensor,
        prediction_steps: int = 48,  # 48 hours
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, np.ndarray]:
        """Generate sequence of predictions"""
        self.eval()
        with torch.no_grad():
            # Initialize sequence and outputs
            sequence = initial_sequence.clone()
            predictions = []
            current_hidden = hidden
            
            # Generate predictions
            for _ in range(prediction_steps):
                # Get prediction
                outputs = self.forward(sequence, current_hidden)
                current_hidden = outputs['hidden']
                
                predictions.append({
                    'eclipse_state': outputs['eclipse_state'].numpy(),
                    'temperature': outputs['temperature'].numpy(),
                    'power_state': outputs['power_state'].numpy(),
                    'uncertainties': outputs['uncertainties'].numpy()
                })
                
                # Update sequence with prediction
                new_step = torch.cat([
                    outputs['eclipse_state'].softmax(dim=-1),
                    outputs['temperature'],
                    outputs['power_state']
                ], dim=-1)
                
                sequence = torch.cat([
                    sequence[:, 1:],
                    new_step.unsqueeze(1)
                ], dim=1)
            
            return predictions
    
    def calculate_eclipse_metrics(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Calculate eclipse-related metrics"""
        # Extract predictions
        eclipse_states = np.array([p['eclipse_state'] for p in predictions])
        temperatures = np.array([p['temperature'] for p in predictions])
        power_states = np.array([p['power_state'] for p in predictions])
        uncertainties = np.array([p['uncertainties'] for p in predictions])
        
        # Calculate eclipse duration
        eclipse_probs = eclipse_states.softmax(axis=-1)
        eclipse_duration = np.sum(eclipse_probs[:, 1:] > 0.5)  # penumbra + umbra
        
        # Calculate thermal metrics
        temp_min = np.min(temperatures)
        temp_max = np.max(temperatures)
        temp_rate = np.mean(np.diff(temperatures, axis=0))
        
        # Calculate power metrics
        battery_min = np.min(power_states[:, 0])
        power_consumption = np.mean(power_states[:, 1])
        
        # Calculate uncertainty metrics
        mean_uncertainty = np.mean(uncertainties, axis=0)
        
        return {
            'eclipse_duration': float(eclipse_duration),
            'temperature_min': float(temp_min),
            'temperature_max': float(temp_max),
            'temperature_rate': float(temp_rate),
            'battery_min': float(battery_min),
            'power_consumption': float(power_consumption),
            'mean_uncertainties': mean_uncertainty.tolist()
        }

def train_eclipse_predictor(
    train_data: torch.Tensor,
    train_labels: Dict[str, torch.Tensor],
    val_data: torch.Tensor,
    val_labels: Dict[str, torch.Tensor],
    model_params: Dict = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[EclipsePredictor, Dict[str, List[float]]]:
    """Train the eclipse prediction model"""
    # Initialize model
    if model_params is None:
        model_params = {
            'input_dim': train_data.shape[-1],
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2
        }
    
    model = EclipsePredictor(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
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
            batch_data = train_data[i:i+batch_size]
            batch_labels = {
                k: v[i:i+batch_size]
                for k, v in train_labels.items()
            }
            
            # Forward pass
            outputs = model(batch_data)
            
            # Calculate losses
            eclipse_loss = ce_loss(
                outputs['eclipse_state'],
                batch_labels['eclipse_state']
            )
            
            thermal_loss = mse_loss(
                outputs['temperature'],
                batch_labels['temperature']
            )
            
            power_loss = mse_loss(
                outputs['power_state'],
                batch_labels['power_state']
            )
            
            # Combined loss
            loss = eclipse_loss + thermal_loss + power_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate accuracy
            eclipse_accuracy = (
                outputs['eclipse_state'].argmax(dim=-1) ==
                batch_labels['eclipse_state']
            ).float().mean()
            train_accuracies.append(eclipse_accuracy.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i+batch_size]
                batch_labels = {
                    k: v[i:i+batch_size]
                    for k, v in val_labels.items()
                }
                
                outputs = model(batch_data)
                
                # Calculate losses
                eclipse_loss = ce_loss(
                    outputs['eclipse_state'],
                    batch_labels['eclipse_state']
                )
                
                thermal_loss = mse_loss(
                    outputs['temperature'],
                    batch_labels['temperature']
                )
                
                power_loss = mse_loss(
                    outputs['power_state'],
                    batch_labels['power_state']
                )
                
                loss = eclipse_loss + thermal_loss + power_loss
                val_losses.append(loss.item())
                
                # Calculate accuracy
                eclipse_accuracy = (
                    outputs['eclipse_state'].argmax(dim=-1) ==
                    batch_labels['eclipse_state']
                ).float().mean()
                val_accuracies.append(eclipse_accuracy.item())
        
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