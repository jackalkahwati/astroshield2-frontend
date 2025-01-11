import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class SignatureHybridNet(nn.Module):
    """Hybrid network for signature recognition"""
    
    def __init__(
        self,
        input_channels: int,  # Now represents feature dimension
        sequence_length: int = 1,
        hidden_size: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal feature extraction (if sequence_length > 1)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, num_classes)
        )
        
        # Feature similarity head
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 128)  # Embedding dimension
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1),
            nn.Softplus()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size, seq_len, features = x.shape
        
        # Process features
        features_list = []
        for t in range(seq_len):
            features = self.feature_net(x[:, t])
            features_list.append(features)
        
        # Stack features
        features = torch.stack(features_list, dim=1)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Generate predictions
        classifications = self.classifier(context)
        embeddings = self.similarity_head(context)
        uncertainties = self.uncertainty_estimator(context)
        
        return {
            'classifications': classifications,
            'embeddings': embeddings,
            'uncertainties': uncertainties,
            'attention_weights': attention_weights
        }
    
    def calculate_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate similarity between two signature embeddings"""
        # Normalize embeddings
        embedding1 = nn.functional.normalize(embedding1, p=2, dim=-1)
        embedding2 = nn.functional.normalize(embedding2, p=2, dim=-1)
        
        # Calculate cosine similarity
        similarity = torch.sum(embedding1 * embedding2, dim=-1)
        return similarity

def train_signature_model(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    val_data: torch.Tensor,
    val_labels: torch.Tensor,
    model_params: Dict = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[SignatureHybridNet, Dict[str, List[float]]]:
    """Train the signature recognition model"""
    # Initialize model
    if model_params is None:
        model_params = {
            'input_channels': train_data.shape[1],
            'sequence_length': train_data.shape[0],
            'hidden_size': 128,
            'num_classes': len(torch.unique(train_labels)),
            'dropout': 0.2
        }
    
    model = SignatureHybridNet(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=0.3)
    
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
            batch_labels = train_labels[i:i+batch_size]
            
            # Forward pass
            predictions = model(batch_data)
            
            # Classification loss
            class_loss = ce_loss(predictions['classifications'], batch_labels)
            
            # Triplet loss for embeddings
            pos_mask = batch_labels.unsqueeze(0) == batch_labels.unsqueeze(1)
            neg_mask = ~pos_mask
            
            embeddings = predictions['embeddings']
            
            # Sample triplets
            anchor_idx = torch.arange(len(embeddings))
            pos_idx = torch.multinomial(pos_mask.float(), 1).squeeze()
            neg_idx = torch.multinomial(neg_mask.float(), 1).squeeze()
            
            triplet = triplet_loss(
                embeddings[anchor_idx],
                embeddings[pos_idx],
                embeddings[neg_idx]
            )
            
            # Combined loss
            loss = class_loss + 0.5 * triplet
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate accuracy
            accuracy = (
                predictions['classifications'].argmax(dim=-1) == batch_labels
            ).float().mean()
            train_accuracies.append(accuracy.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i+batch_size]
                batch_labels = val_labels[i:i+batch_size]
                
                predictions = model(batch_data)
                
                # Calculate losses
                class_loss = ce_loss(predictions['classifications'], batch_labels)
                
                val_losses.append(class_loss.item())
                
                # Calculate accuracy
                accuracy = (
                    predictions['classifications'].argmax(dim=-1) == batch_labels
                ).float().mean()
                val_accuracies.append(accuracy.item())
        
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