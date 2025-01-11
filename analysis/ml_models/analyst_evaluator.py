import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class AnalystConsensus:
    consensus_reached: bool
    confidence: float
    recommended_classification: str
    supporting_evidence: List[str]
    dissenting_opinions: List[Dict]
    timestamp: datetime

class ConsensusNet(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Feature extraction for analyst inputs
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Self-attention mechanism for analyst opinions
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Classification consensus head
        self.consensus_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10)  # Number of possible classifications
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Evidence weighting head
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # Number of evidence types
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply self-attention
        features = features.unsqueeze(0)  # Add sequence dimension
        attended_features, _ = self.attention(features, features, features)
        features = attended_features.squeeze(0)  # Remove sequence dimension
        
        # Generate outputs
        consensus_logits = self.consensus_head(features)
        confidence = self.confidence_head(features)
        evidence_weights = torch.sigmoid(self.evidence_head(features))
        
        return consensus_logits, confidence, evidence_weights

class AnalystEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'input_dim': 128,
            'hidden_dim': 256,
            'confidence_threshold': 0.75,
            'min_consensus_threshold': 0.7
        }
        
        self.model = ConsensusNet(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim']
        )
        
        self.classifications = [
            'MILITARY',
            'COMMERCIAL',
            'CIVIL',
            'EXPERIMENTAL',
            'DEBRIS',
            'UNKNOWN',
            'MANEUVERABLE',
            'NON_MANEUVERABLE',
            'ACTIVE',
            'INACTIVE'
        ]
        
        self.evidence_types = [
            'ORBITAL_BEHAVIOR',
            'RF_SIGNATURE',
            'PHYSICAL_CHARACTERISTICS',
            'LAUNCH_DATA',
            'HISTORICAL_PATTERNS',
            'REGISTRATION_INFO',
            'MISSION_STATEMENTS',
            'TECHNICAL_CAPABILITIES'
        ]

    def _preprocess_data(self, analyst_inputs: List[Dict]) -> torch.Tensor:
        """Convert analyst inputs to model input format"""
        features = []
        
        for analyst_input in analyst_inputs:
            # Analyst classification one-hot encoding
            classification = analyst_input.get('classification', 'UNKNOWN')
            class_idx = self.classifications.index(classification)
            class_encoding = [0] * len(self.classifications)
            class_encoding[class_idx] = 1
            features.extend(class_encoding)
            
            # Confidence score
            features.append(analyst_input.get('confidence', 0.5))
            
            # Evidence weights
            evidence = analyst_input.get('evidence', {})
            for evidence_type in self.evidence_types:
                features.append(evidence.get(evidence_type, 0))
        
        # Pad to match input dimension
        while len(features) < self.config['input_dim']:
            features.append(0)
        features = features[:self.config['input_dim']]
        
        return torch.FloatTensor(features)

    def _get_supporting_evidence(self, evidence_weights: torch.Tensor) -> List[str]:
        """Get supporting evidence types based on weights"""
        evidence = []
        weights = evidence_weights.numpy()
        for i, weight in enumerate(weights):
            if weight > self.config['confidence_threshold']:
                evidence.append(self.evidence_types[i])
        return evidence

    def _get_dissenting_opinions(
        self,
        analyst_inputs: List[Dict],
        consensus: str
    ) -> List[Dict]:
        """Identify dissenting opinions"""
        dissent = []
        for analyst_input in analyst_inputs:
            if analyst_input['classification'] != consensus:
                dissent.append({
                    'classification': analyst_input['classification'],
                    'confidence': analyst_input['confidence'],
                    'evidence': analyst_input.get('evidence', {})
                })
        return dissent

    async def evaluate_consensus(self, analyst_inputs: List[Dict]) -> AnalystConsensus:
        """
        Evaluate analyst consensus using the ML model
        """
        try:
            # Preprocess input data
            x = self._preprocess_data(analyst_inputs)
            
            # Model inference
            with torch.no_grad():
                consensus_logits, confidence, evidence_weights = self.model(x.unsqueeze(0))
                
                # Get consensus classification
                consensus_probs = torch.softmax(consensus_logits[0], dim=0)
                max_prob, max_idx = torch.max(consensus_probs, dim=0)
                consensus_reached = max_prob > self.config['min_consensus_threshold']
                recommended_class = self.classifications[max_idx]
                
                # Get supporting evidence and dissenting opinions
                supporting_evidence = self._get_supporting_evidence(evidence_weights[0])
                dissenting_opinions = self._get_dissenting_opinions(
                    analyst_inputs,
                    recommended_class
                )
                
                return AnalystConsensus(
                    consensus_reached=consensus_reached,
                    confidence=float(confidence[0]),
                    recommended_classification=recommended_class,
                    supporting_evidence=supporting_evidence,
                    dissenting_opinions=dissenting_opinions,
                    timestamp=datetime.now()
                )

        except Exception as e:
            print(f"Error in consensus evaluation: {str(e)}")
            return None

    def export_to_onnx(self, path: str):
        """Export model to ONNX format"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.config['input_dim'])
            
            # Export the model
            torch.onnx.export(
                self.model,
                dummy_input,
                path,
                export_params=True,
                opset_version=13,  # Updated to version 13 for unflatten support
                do_constant_folding=True,
                input_names=['input'],
                output_names=['consensus_logits', 'confidence', 'evidence_weights'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'consensus_logits': {0: 'batch_size'},
                    'confidence': {0: 'batch_size'},
                    'evidence_weights': {0: 'batch_size'}
                }
            )
            
            print(f"Model exported to: {path}")
            
            # Verify the exported model
            import onnx
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed")
            
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
