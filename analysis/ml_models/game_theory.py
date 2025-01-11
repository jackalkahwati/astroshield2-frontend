import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DeceptionStrategy:
    type: str
    parameters: Dict
    expected_effectiveness: float
    resource_cost: float
    duration: float

@dataclass
class AdversaryModel:
    behavior_type: str
    capabilities: Dict
    estimated_knowledge: Dict
    response_patterns: List[Dict]

@dataclass
class GameTheoryResult:
    optimal_strategy: DeceptionStrategy
    expected_payoff: float
    adversary_prediction: Dict
    confidence: float
    adaptation_path: List[Dict]

class AdversaryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class StrategyGenerator(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, strategy_dim: int = 64):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, strategy_dim)
        )
        
        # Strategy parameters head
        self.param_head = nn.Sequential(
            nn.Linear(strategy_dim, strategy_dim // 2),
            nn.ReLU(),
            nn.Linear(strategy_dim // 2, 3)  # num_layers, complexity, synchronization
        )
        
        # Effectiveness prediction head
        self.effectiveness_head = nn.Sequential(
            nn.Linear(strategy_dim, strategy_dim // 2),
            nn.ReLU(),
            nn.Linear(strategy_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state, adversary_encoding):
        combined = torch.cat([state, adversary_encoding], dim=-1)
        strategy_features = self.generator(combined)
        parameters = torch.sigmoid(self.param_head(strategy_features))
        effectiveness = self.effectiveness_head(strategy_features)
        return parameters, effectiveness

class GameTheoryDeception:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'num_strategies': 5,
            'num_adversary_actions': 5,
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'state_dim': 32,
            'adversary_dim': 16,
            'hidden_dim': 128,
            'strategy_dim': 64
        }
        
        # Initialize neural networks
        self.adversary_encoder = AdversaryEncoder(
            input_dim=self.config['adversary_dim'],
            hidden_dim=self.config['hidden_dim']
        )
        
        self.strategy_generator = StrategyGenerator(
            state_dim=self.config['state_dim'],
            hidden_dim=self.config['hidden_dim'],
            strategy_dim=self.config['strategy_dim']
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.adversary_encoder.parameters()) +
            list(self.strategy_generator.parameters()),
            lr=self.config['learning_rate']
        )

    def _preprocess_input(
        self,
        current_state: Dict,
        adversary_observations: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert input data to model format"""
        # Process state information
        state_features = []
        
        # System state features
        state_features.extend([
            current_state.get('resource_level', 0),
            current_state.get('vulnerability_index', 0),
            current_state.get('detection_probability', 0)
        ])
        
        # Current defense posture
        defense = current_state.get('defense_posture', {})
        state_features.extend([
            defense.get('encryption_level', 0),
            defense.get('firewall_strength', 0),
            defense.get('deception_complexity', 0)
        ])
        
        # Pad state features
        while len(state_features) < self.config['state_dim']:
            state_features.append(0)
        state_features = state_features[:self.config['state_dim']]
        
        # Process adversary observations
        adversary_features = []
        
        if adversary_observations:
            latest_obs = adversary_observations[0]
            adversary_features.extend([
                latest_obs.get('attack_sophistication', 0),
                latest_obs.get('persistence_level', 0),
                latest_obs.get('resource_commitment', 0)
            ])
        
        # Pad adversary features
        while len(adversary_features) < self.config['adversary_dim']:
            adversary_features.append(0)
        adversary_features = adversary_features[:self.config['adversary_dim']]
        
        return (
            torch.FloatTensor(state_features),
            torch.FloatTensor(adversary_features)
        )

    def _create_deception_strategy(
        self,
        parameters: torch.Tensor,
        effectiveness: float
    ) -> DeceptionStrategy:
        """Convert model output to deception strategy"""
        params = parameters.detach().numpy()
        
        return DeceptionStrategy(
            type='MULTI_LAYER_DECEPTION',
            parameters={
                'num_layers': int(params[0] * 5) + 1,  # 1-5 layers
                'complexity': float(params[1]),
                'synchronization': float(params[2])
            },
            expected_effectiveness=float(effectiveness),
            resource_cost=float(params[1] * 0.8),  # cost scales with complexity
            duration=7200.0  # 2 hours in seconds
        )

    async def optimize_strategy(
        self,
        current_state: Dict,
        adversary_observations: List[Dict]
    ) -> GameTheoryResult:
        """
        Generate optimal deception strategy using MARL model
        """
        try:
            # Preprocess input data
            state_tensor, adversary_tensor = self._preprocess_input(
                current_state,
                adversary_observations
            )
            
            # Model inference
            with torch.no_grad():
                # Encode adversary behavior
                adversary_encoding = self.adversary_encoder(adversary_tensor.unsqueeze(0))
                
                # Generate strategy
                parameters, effectiveness = self.strategy_generator(
                    state_tensor.unsqueeze(0),
                    adversary_encoding
                )
            
            # Create deception strategy
            strategy = self._create_deception_strategy(
                parameters[0],
                effectiveness.item()
            )
            
            # Predict adversary response
            adversary_prediction = {
                'expected_response': 'INCREASED_SURVEILLANCE',
                'probability': float(effectiveness.item()),
                'time_to_response': 300
            }
            
            # Generate adaptation path
            adaptation_path = [
                {
                    'step': 0,
                    'parameters': strategy.parameters,
                    'expected_effectiveness': strategy.expected_effectiveness
                }
            ]
            
            return GameTheoryResult(
                optimal_strategy=strategy,
                expected_payoff=float(effectiveness.item()),
                adversary_prediction=adversary_prediction,
                confidence=float(effectiveness.item()),
                adaptation_path=adaptation_path
            )

        except Exception as e:
            print(f"Error in strategy optimization: {str(e)}")
            return None

    def export_to_onnx(self, path: str):
        """Export model to ONNX format"""
        try:
            # Create dummy inputs
            dummy_state = torch.randn(1, self.config['state_dim'])
            dummy_adversary = torch.randn(1, self.config['adversary_dim'])
            
            # Create a wrapper class for ONNX export
            class ModelWrapper(nn.Module):
                def __init__(self, adversary_encoder, strategy_generator):
                    super().__init__()
                    self.adversary_encoder = adversary_encoder
                    self.strategy_generator = strategy_generator
                
                def forward(self, state, adversary):
                    adversary_encoding = self.adversary_encoder(adversary)
                    parameters, effectiveness = self.strategy_generator(
                        state,
                        adversary_encoding
                    )
                    return parameters, effectiveness
            
            wrapper = ModelWrapper(self.adversary_encoder, self.strategy_generator)
            
            # Export the model
            torch.onnx.export(
                wrapper,
                (dummy_state, dummy_adversary),
                path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['state', 'adversary'],
                output_names=['parameters', 'effectiveness'],
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    'adversary': {0: 'batch_size'},
                    'parameters': {0: 'batch_size'},
                    'effectiveness': {0: 'batch_size'}
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
