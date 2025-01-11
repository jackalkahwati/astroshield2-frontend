import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from typing import Dict, List, Tuple, Optional

class TrackEncoder(nn.Module):
    """Encodes track features into a latent representation"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        
        # LSTM for processing track sequences
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # MLP for final encoding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take final hidden state
        final_hidden = lstm_out[:, -1]
        
        # Project to latent space
        latent = self.mlp(final_hidden)
        return latent

class EdgePredictor(nn.Module):
    """Predicts if two tracks belong to the same object"""
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass"""
        # Get node pairs for each edge
        src, dst = edge_index
        
        # Concatenate node features for each edge
        edge_features = torch.cat([
            node_features[src],
            node_features[dst]
        ], dim=1)
        
        # Predict edge probabilities
        edge_pred = self.edge_mlp(edge_features)
        return edge_pred

class TrackGNN(nn.Module):
    """Graph Neural Network for track association"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Track encoder
        self.encoder = TrackEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GCNConv(
                in_channels=latent_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim
            ) for i in range(num_layers)
        ])
        
        # Edge predictor
        self.edge_predictor = EdgePredictor(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # Track classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 4)  # 4 track types
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus()
        )
    
    def forward(
        self,
        track_sequences: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode track sequences
        node_features = self.encoder(track_sequences)
        
        # Apply graph convolutions
        x = node_features
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Predict edges
        edge_pred = self.edge_predictor(x, edge_index)
        
        # Classify tracks
        track_classes = self.classifier(x)
        
        # Estimate uncertainties
        uncertainties = self.uncertainty(x)
        
        return {
            'node_features': x,
            'edge_pred': edge_pred,
            'track_classes': track_classes,
            'uncertainties': uncertainties
        }
    
    def associate_tracks(
        self,
        track_sequences: torch.Tensor,
        edge_index: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Associate tracks into groups"""
        self.eval()
        with torch.no_grad():
            # Get predictions
            outputs = self.forward(track_sequences, edge_index)
            edge_probs = outputs['edge_pred']
            
            # Create adjacency matrix
            n = len(track_sequences)
            adj_matrix = torch.zeros((n, n))
            edge_index_tuple = tuple(edge_index.cpu().numpy())
            adj_matrix[edge_index_tuple] = (edge_probs > threshold).float().cpu()
            
            # Find connected components (track groups)
            groups = []
            visited = set()
            
            for i in range(n):
                if i not in visited:
                    group = []
                    stack = [i]
                    
                    while stack:
                        node = stack.pop()
                        if node not in visited:
                            visited.add(node)
                            group.append(node)
                            
                            # Add neighbors
                            neighbors = torch.where(adj_matrix[node] > 0)[0]
                            stack.extend(
                                neighbor.item()
                                for neighbor in neighbors
                                if neighbor.item() not in visited
                            )
                    
                    groups.append(group)
            
            return groups, outputs['uncertainties']

def train_track_gnn(
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model_params: Dict = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[TrackGNN, Dict[str, List[float]]]:
    """Train the track association model"""
    # Unpack data
    train_sequences, train_edge_index, train_labels = train_data
    val_sequences, val_edge_index, val_labels = val_data
    
    # Initialize model
    if model_params is None:
        model_params = {
            'input_dim': train_sequences.shape[-1],
            'hidden_dim': 128,
            'latent_dim': 64,
            'num_layers': 3
        }
    
    model = TrackGNN(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    edge_loss_fn = nn.BCELoss()
    class_loss_fn = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_edge_acc': [],
        'val_edge_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        
        # Training
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i+batch_size]
            batch_edge_index = train_edge_index[:, train_edge_index[0] < i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_sequences, batch_edge_index)
            
            # Calculate losses
            edge_loss = edge_loss_fn(
                outputs['edge_pred'],
                batch_labels[batch_edge_index[0], batch_edge_index[1]]
            )
            
            class_loss = class_loss_fn(
                outputs['track_classes'],
                batch_labels
            )
            
            # Combined loss
            loss = edge_loss + 0.5 * class_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate accuracy
            edge_accuracy = (
                (outputs['edge_pred'] > 0.5).float() ==
                batch_labels[batch_edge_index[0], batch_edge_index[1]]
            ).float().mean()
            train_accuracies.append(edge_accuracy.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for i in range(0, len(val_sequences), batch_size):
                batch_sequences = val_sequences[i:i+batch_size]
                batch_edge_index = val_edge_index[:, val_edge_index[0] < i+batch_size]
                batch_labels = val_labels[i:i+batch_size]
                
                outputs = model(batch_sequences, batch_edge_index)
                
                # Calculate losses
                edge_loss = edge_loss_fn(
                    outputs['edge_pred'],
                    batch_labels[batch_edge_index[0], batch_edge_index[1]]
                )
                
                class_loss = class_loss_fn(
                    outputs['track_classes'],
                    batch_labels
                )
                
                loss = edge_loss + 0.5 * class_loss
                val_losses.append(loss.item())
                
                # Calculate accuracy
                edge_accuracy = (
                    (outputs['edge_pred'] > 0.5).float() ==
                    batch_labels[batch_edge_index[0], batch_edge_index[1]]
                ).float().mean()
                val_accuracies.append(edge_accuracy.item())
        
        # Update history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_edge_acc'].append(np.mean(train_accuracies))
        history['val_edge_acc'].append(np.mean(val_accuracies))
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"Train Edge Acc: {history['train_edge_acc'][-1]:.4f}")
            print(f"Val Edge Acc: {history['val_edge_acc'][-1]:.4f}")
    
    return model, history 