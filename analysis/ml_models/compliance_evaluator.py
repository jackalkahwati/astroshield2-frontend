import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ComplianceResult:
    is_compliant: bool
    confidence: float
    violations: List[str]
    regulation_refs: List[str]
    timestamp: datetime

class ComplianceNet(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Compliance classification head
        self.compliance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Violation type classification head
        self.violation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # 5 violation types
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        compliance_prob = self.compliance_head(features)
        violation_logits = self.violation_head(features)
        confidence = self.confidence_head(features)
        return compliance_prob, violation_logits, confidence

class ComplianceEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'input_dim': 64,
            'hidden_dim': 128,
            'confidence_threshold': 0.75
        }
        
        self.model = ComplianceNet(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim']
        )
        
        self.violation_types = [
            'ITU_VIOLATION',
            'FCC_VIOLATION',
            'UN_REGISTRY_MISSING',
            'FREQUENCY_VIOLATION',
            'ORBITAL_SLOT_VIOLATION'
        ]
        
        self.regulation_refs = {
            'ITU_VIOLATION': ['ITU-R S.1503', 'ITU-R S.1428'],
            'FCC_VIOLATION': ['47 CFR §25', '47 CFR §97'],
            'UN_REGISTRY_MISSING': ['UNOOSA Reg. Article VI'],
            'FREQUENCY_VIOLATION': ['ITU-R Article 5', 'FCC Part 97.207'],
            'ORBITAL_SLOT_VIOLATION': ['ITU-R Appendix 30B']
        }

    def _preprocess_data(self, satellite_data: Dict) -> torch.Tensor:
        """Convert satellite data to model input format"""
        features = []
        
        # Orbital parameters
        orbit = satellite_data.get('orbit', {})
        features.extend([
            orbit.get('semi_major_axis', 0),
            orbit.get('eccentricity', 0),
            orbit.get('inclination', 0),
            orbit.get('raan', 0),
            orbit.get('arg_perigee', 0),
            orbit.get('mean_anomaly', 0)
        ])
        
        # Frequency information
        freq = satellite_data.get('frequency', {})
        features.extend([
            freq.get('uplink', 0),
            freq.get('downlink', 0),
            freq.get('bandwidth', 0)
        ])
        
        # Registration information
        reg = satellite_data.get('registration', {})
        features.extend([
            float(reg.get('has_itu_filing', False)),
            float(reg.get('has_fcc_license', False)),
            float(reg.get('in_un_registry', False))
        ])
        
        # Pad to match input dimension
        while len(features) < self.config['input_dim']:
            features.append(0)
        features = features[:self.config['input_dim']]
        
        return torch.FloatTensor(features)

    def _get_violations(self, violation_logits: torch.Tensor) -> List[str]:
        """Convert violation logits to violation types"""
        probs = torch.sigmoid(violation_logits)
        violations = []
        for i, prob in enumerate(probs):
            if prob > self.config['confidence_threshold']:
                violations.append(self.violation_types[i])
        return violations

    def _get_regulation_refs(self, violations: List[str]) -> List[str]:
        """Get relevant regulation references for violations"""
        refs = []
        for violation in violations:
            refs.extend(self.regulation_refs.get(violation, []))
        return list(set(refs))  # Remove duplicates

    async def evaluate_compliance(self, satellite_data: Dict) -> ComplianceResult:
        """
        Evaluate satellite compliance using the ML model
        """
        try:
            # Preprocess input data
            x = self._preprocess_data(satellite_data)
            
            # Model inference
            with torch.no_grad():
                compliance_prob, violation_logits, confidence = self.model(x.unsqueeze(0))
                
                is_compliant = bool(compliance_prob[0] > self.config['confidence_threshold'])
                violations = self._get_violations(violation_logits[0])
                regulation_refs = self._get_regulation_refs(violations)
                
                return ComplianceResult(
                    is_compliant=is_compliant,
                    confidence=float(confidence[0]),
                    violations=violations,
                    regulation_refs=regulation_refs,
                    timestamp=datetime.now()
                )

        except Exception as e:
            print(f"Error in compliance evaluation: {str(e)}")
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
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['compliance_prob', 'violation_logits', 'confidence'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'compliance_prob': {0: 'batch_size'},
                    'violation_logits': {0: 'batch_size'},
                    'confidence': {0: 'batch_size'}
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
