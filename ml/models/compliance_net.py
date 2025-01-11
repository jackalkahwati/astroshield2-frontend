import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class ComplianceNet(nn.Module):
    def __init__(self,
                 input_dim: int = 15,
                 hidden_dim: int = 128,
                 num_rules: int = 32,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """Initialize the compliance network model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_rules: Number of compliance rules to evaluate
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rules = num_rules
        
        # Input projection with gradual dimension increase
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Learnable rule embeddings
        self.rule_embeddings = nn.Parameter(
            torch.randn(num_rules, hidden_dim)
        )
        
        # Transformer layers for rule evaluation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Rule-specific compliance heads
        self.rule_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_rules)
        ])
        
        # Overall compliance score head
        self.compliance_head = nn.Sequential(
            nn.Linear(num_rules, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Violation severity assessment head
        self.severity_head = nn.Sequential(
            nn.Linear(num_rules, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rules),
            nn.Softplus()  # Non-negative severity scores
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def evaluate_rule_compliance(self,
                               encoded_state: torch.Tensor,
                               rule_idx: int) -> torch.Tensor:
        """Evaluate compliance with a specific rule.
        
        Args:
            encoded_state: Encoded state tensor
            rule_idx: Index of the rule to evaluate
            
        Returns:
            Compliance score for the rule
        """
        rule_embedding = self.rule_embeddings[rule_idx].unsqueeze(0)
        rule_embedding = rule_embedding.expand(encoded_state.shape[0], -1)
        
        # Concatenate state and rule embeddings
        combined = torch.cat([encoded_state, rule_embedding], dim=-1)
        
        # Get compliance score
        compliance_score = self.rule_heads[rule_idx](combined)
        return compliance_score
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing compliance analysis
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)
        
        # Add rule embeddings to sequence
        rules = self.rule_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([x, rules], dim=1)
        
        # Apply transformer
        encoded = self.transformer(x)
        
        # Split encoded sequence back into state and rules
        encoded_state = encoded[:, 0]  # Use first token as state encoding
        
        # Evaluate compliance for each rule
        rule_scores = []
        for i in range(self.num_rules):
            score = self.evaluate_rule_compliance(encoded_state, i)
            rule_scores.append(score)
        
        rule_scores = torch.cat(rule_scores, dim=-1)
        
        # Calculate overall compliance and severity
        overall_compliance = self.compliance_head(rule_scores)
        violation_severity = self.severity_head(rule_scores)
        
        return {
            "rule_compliance": rule_scores,
            "overall_compliance": overall_compliance,
            "violation_severity": violation_severity,
            "encoded_state": encoded_state
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make compliance predictions for input data.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing compliance predictions and analysis
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Threshold compliance scores
            is_compliant = outputs["rule_compliance"] > 0.5
            
            predictions = {
                "rule_compliance_scores": outputs["rule_compliance"],
                "is_compliant": is_compliant,
                "overall_compliance_score": outputs["overall_compliance"],
                "violation_severity": outputs["violation_severity"],
                "high_severity_violations": outputs["violation_severity"] > 1.0
            }
            
            return predictions
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path)) 