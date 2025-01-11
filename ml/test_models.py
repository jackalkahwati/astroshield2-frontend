import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ml.data_generation.synthetic_data import SyntheticDataGenerator
from ml.models.adversary_encoder import AdversaryEncoder
from ml.models.strategy_generator import StrategyGenerator
from ml.models.actor_critic import ActorCritic
from ml.models.threat_detector import ThreatDetectorNN
from ml.utils.metrics import ModelMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, models_dir: str, results_dir: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir) if results_dir else Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator()
        self.metrics = ModelMetrics()
        
        # Load models
        self.models = self._load_models()
    
    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all trained models."""
        models = {}
        
        # Load Adversary Encoder
        adversary_path = self.models_dir / "adversary_encoder.pt"
        if adversary_path.exists():
            models["adversary_encoder"] = AdversaryEncoder()
            models["adversary_encoder"].load_model(str(adversary_path))
        
        # Load Strategy Generator
        strategy_path = self.models_dir / "strategy_generator.pt"
        if strategy_path.exists():
            models["strategy_generator"] = StrategyGenerator()
            models["strategy_generator"].load_model(str(strategy_path))
        
        # Load Actor Critic
        actor_critic_path = self.models_dir / "actor_critic.pt"
        if actor_critic_path.exists():
            models["actor_critic"] = ActorCritic()
            models["actor_critic"].load_model(str(actor_critic_path))
        
        # Load Threat Detector
        threat_detector_path = self.models_dir / "threat_detector.pt"
        if threat_detector_path.exists():
            models["threat_detector"] = ThreatDetectorNN()
            models["threat_detector"].load_model(str(threat_detector_path))
        
        return models
    
    def test_adversary_encoder(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Test the adversary encoder model."""
        logger.info("Testing Adversary Encoder...")
        model = self.models.get("adversary_encoder")
        if not model:
            logger.warning("Adversary Encoder model not found")
            return {}
        
        # Generate test data
        test_data, test_labels = self.data_generator.generate_adversarial_data(num_samples)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            encoded, decoded = model(test_data)
            anomaly_results = model.detect_anomalies(test_data)
        
        # Calculate metrics
        metrics = {
            "reconstruction_error": self.metrics.mse_loss(decoded, test_data).item(),
            "anomaly_detection_accuracy": (
                (anomaly_results["is_anomaly"].float() == test_labels).float().mean().item()
            )
        }
        
        return {
            "metrics": metrics,
            "predictions": anomaly_results
        }
    
    def test_strategy_generator(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Test the strategy generator model."""
        logger.info("Testing Strategy Generator...")
        model = self.models.get("strategy_generator")
        if not model:
            logger.warning("Strategy Generator model not found")
            return {}
        
        # Generate test scenarios
        test_states = torch.randn(num_episodes, 64)  # 64-dimensional state
        
        # Evaluate model
        model.eval()
        strategies = []
        metrics = {
            "mean_value": 0.0,
            "mean_entropy": 0.0
        }
        
        with torch.no_grad():
            for state in test_states:
                strategy = model.generate_strategy(state.unsqueeze(0), deterministic=True)
                strategies.append(strategy)
                
                # Calculate metrics
                metrics["mean_value"] += float(strategy["state_value"].mean())
                action_probs = strategy["action_probs"]
                entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
                metrics["mean_entropy"] += float(entropy)
        
        # Average metrics
        metrics = {k: v / num_episodes for k, v in metrics.items()}
        
        return {
            "metrics": metrics,
            "strategies": strategies
        }
    
    def test_actor_critic(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Test the actor-critic model."""
        logger.info("Testing Actor-Critic Model...")
        model = self.models.get("actor_critic")
        if not model:
            logger.warning("Actor-Critic model not found")
            return {}
        
        # Generate test data
        state_data = self.data_generator.generate_spacecraft_state(1000)
        state_columns = [
            'position_x', 'position_y', 'position_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
            'fuel_remaining'
        ]
        states = torch.tensor(state_data[state_columns].values, dtype=torch.float32)
        
        # Evaluate model
        model.eval()
        metrics = {
            "mean_value": 0.0,
            "mean_action_magnitude": 0.0,
            "mean_reward": 0.0
        }
        actions = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                state = states[np.random.randint(len(states))]
                action, info = model.select_action(state.unsqueeze(0), deterministic=True)
                value = model.compute_value(state.unsqueeze(0))
                
                metrics["mean_value"] += float(value.mean())
                metrics["mean_action_magnitude"] += float(torch.norm(action.delta_v))
                metrics["mean_reward"] += float(model._compute_action_log_probs(action, action).mean())
                
                actions.append({
                    "delta_v": action.delta_v.numpy(),
                    "thrust_duration": float(action.thrust_duration.mean()),
                    "orientation": action.orientation.numpy()
                })
        
        # Average metrics
        metrics = {k: v / num_episodes for k, v in metrics.items()}
        
        return {
            "metrics": metrics,
            "actions": actions
        }
    
    def test_threat_detector(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Test the threat detector model."""
        logger.info("Testing Threat Detector...")
        model = self.models.get("threat_detector")
        if not model:
            logger.warning("Threat Detector model not found")
            return {}
        
        # Generate test data
        test_data, test_labels, risk_factors = self.data_generator.generate_threat_data(num_samples)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            assessment = model.detect_threats(test_data)
            risk_assessment = model.assess_risk(test_data)
        
        # Calculate metrics
        metrics = self.metrics.get_all_metrics(
            assessment.threat_type.unsqueeze(1),
            test_labels,
            task_type="classification"
        )
        
        # Add risk assessment metrics
        for factor, value in risk_assessment.items():
            if isinstance(value, torch.Tensor):
                metrics[f"mean_{factor}"] = float(value.mean())
        
        return {
            "metrics": metrics,
            "predictions": {
                "threat_level": assessment.threat_level.numpy(),
                "threat_type": assessment.threat_type.numpy(),
                "confidence": assessment.confidence.numpy(),
                "risk_factors": {k: v.numpy() for k, v in assessment.risk_factors.items()}
            }
        }
    
    def test_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Test all models and return results."""
        results = {}
        
        # Test each model
        if "adversary_encoder" in self.models:
            results["adversary_encoder"] = self.test_adversary_encoder()
        
        if "strategy_generator" in self.models:
            results["strategy_generator"] = self.test_strategy_generator()
        
        if "actor_critic" in self.models:
            results["actor_critic"] = self.test_actor_critic()
        
        if "threat_detector" in self.models:
            results["threat_detector"] = self.test_threat_detector()
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, Any]]):
        """Save test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_results_path = self.results_dir / f"test_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "metrics": model_results.get("metrics", {}),
                "predictions": self._convert_to_serializable(model_results.get("predictions", {}))
            }
        
        with open(test_results_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Test results saved to {test_results_path}")
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and tensors to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

if __name__ == "__main__":
    # Assuming models are saved in the "checkpoints/latest" directory
    latest_models_dir = Path("checkpoints/latest")
    
    # Initialize and run tests
    tester = ModelTester(latest_models_dir)
    results = tester.test_all_models()
    
    # Print summary
    print("\nTest Results Summary:")
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        if "metrics" in model_results:
            for metric, value in model_results["metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}") 