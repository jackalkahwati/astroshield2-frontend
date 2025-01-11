import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from ml.models.adversary_encoder import AdversaryEncoder
from ml.models.strategy_generator import StrategyGenerator
from ml.models.actor_critic import ActorCritic
from ml.models.threat_detector import ThreatDetectorNN
from ml.data_generation.synthetic_data import SyntheticDataGenerator
from ml.utils.trainer import ModelTrainer
from ml.utils.metrics import ModelMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    def __init__(self):
        """Initialize the training pipeline."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_generator = SyntheticDataGenerator()
        self.metrics = ModelMetrics()
        self.base_path = Path("checkpoints")
        
        # Load configurations
        self.configs = {
            "adversary_encoder": {
                "batch_size": 32,
                "learning_rate": 1e-3,
                "num_epochs": 50,
                "input_dim": 128,
                "encoding_dim": 64
            },
            "strategy_generator": {
                "batch_size": 32,
                "learning_rate": 3e-4,
                "num_episodes": 1000,
                "gamma": 0.99,
                "state_dim": 14,
                "action_dim": 6,
                "hidden_dim": 128
            },
            "actor_critic": {
                "batch_size": 32,
                "learning_rate": 3e-4,
                "num_episodes": 1000,
                "gamma": 0.99,
                "state_dim": 14,
                "action_dim": 6,
                "hidden_dim": 128
            },
            "threat_detector": {
                "batch_size": 32,
                "learning_rate": 1e-3,
                "num_epochs": 50,
                "input_dim": 64,
                "hidden_dim": 128,
                "num_classes": 5
            }
        }
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Initialize metrics
        self.metrics = ModelMetrics()
    
    def train_adversary_encoder(self) -> Dict[str, Any]:
        """Train the adversary encoder model."""
        logger.info("Training Adversary Encoder...")
        
        # Generate synthetic data
        data = self.data_generator.generate_adversarial_data(10000)
        data_tensor = torch.FloatTensor(data)  # Convert to tensor
        dataset = TensorDataset(data_tensor, data_tensor)  # Use same data for input and target (autoencoder)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.configs["adversary_encoder"]["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.configs["adversary_encoder"]["batch_size"]
        )
        
        # Initialize model and training components
        model = AdversaryEncoder().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.configs["adversary_encoder"]["learning_rate"]
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            verbose=True
        )
        
        # Create checkpoint directory
        os.makedirs("checkpoints/adversary_encoder", exist_ok=True)
        
        # Setup trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            config={
                "num_epochs": self.configs["adversary_encoder"]["num_epochs"],
                "checkpoint_path": "checkpoints/adversary_encoder/adversary_encoder_best.pt"
            },
            device=self.device
        )
        
        # Train the model
        history = trainer.train()
        
        # Save the model
        torch.save(model.state_dict(), "checkpoints/adversary_encoder/adversary_encoder_final.pt")
        
        return history
    
    def train_strategy_generator(self) -> Dict[str, Any]:
        """Train the strategy generator model."""
        logger.info("Training Strategy Generator...")
        
        try:
            # Initialize model and training components
            model = StrategyGenerator(
                state_dim=self.configs["strategy_generator"]["state_dim"],
                action_dim=self.configs["strategy_generator"]["action_dim"]
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.configs["strategy_generator"]["learning_rate"])
            
            # Training loop
            num_episodes = self.configs["strategy_generator"]["num_episodes"]
            gamma = self.configs["strategy_generator"]["gamma"]
            
            for episode in range(1, num_episodes + 1):
                state = self.data_generator.generate_spacecraft_state()
                done = False
                total_reward = 0
                policy_loss = 0
                value_loss = 0
                
                while not done:
                    # Ensure proper tensor shape (batch_size, state_dim)
                    if isinstance(state, np.ndarray):
                        state = torch.FloatTensor(state)
                    if state.dim() == 1:
                        state = state.unsqueeze(0)
                    state_tensor = state.to(self.device)
                    
                    # Get action and value from model
                    action_mean, value = model(state_tensor)
                    
                    # Add exploration noise
                    noise = torch.randn_like(action_mean) * 0.1
                    action = torch.clamp(action_mean + noise, -1.0, 1.0)
                    
                    # Take action in environment
                    next_state, reward, done = self.data_generator.step(state_tensor, action)
                    
                    # Ensure proper tensor shapes for reward and next_state
                    if isinstance(reward, (float, int)):
                        reward_tensor = torch.FloatTensor([[reward]]).to(self.device)
                    elif isinstance(reward, np.ndarray):
                        reward_tensor = torch.FloatTensor(reward.reshape(-1, 1)).to(self.device)
                    else:
                        reward_tensor = torch.FloatTensor([[float(reward)]]).to(self.device)
                    
                    if isinstance(next_state, np.ndarray):
                        next_state = torch.FloatTensor(next_state)
                    if next_state.dim() == 1:
                        next_state = next_state.unsqueeze(0)
                    next_state = next_state.to(self.device)
                    
                    # Get next value
                    _, next_value = model(next_state)
                    next_value = next_value.detach()
                    
                    # Compute losses with proper shapes
                    value_loss = F.mse_loss(value, reward_tensor + gamma * next_value * (1 - float(done)))
                    
                    # Calculate policy loss with proper dimension handling
                    advantages = (reward_tensor + gamma * next_value * (1 - float(done))) - value.detach()
                    action_log_probs = torch.log_softmax(action_mean, dim=-1)
                    policy_loss = -(action_log_probs * advantages).mean()
                    
                    # Update model
                    loss = policy_loss + value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update state and accumulate reward
                    state = next_state
                    total_reward += reward_tensor.item()
                
                # Log progress
                if episode % 10 == 0:
                    logger.info(f"Episode {episode}/{num_episodes} - "
                              f"policy_loss: {policy_loss:.4f} - "
                              f"value_loss: {value_loss:.4f} - "
                              f"reward: {total_reward:.4f} - "
                              f"total_loss: {loss.item():.4f}")
            
            # Save the model
            os.makedirs("checkpoints/strategy_generator", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/strategy_generator/strategy_generator_final.pt")
            logger.info("Saved strategy generator model")
            
            return {"final_reward": total_reward}
            
        except Exception as e:
            logger.error(f"Error training Strategy Generator: {str(e)}")
            raise
    
    def train_actor_critic(self) -> Dict[str, Any]:
        """Train the actor-critic model."""
        logger.info("Training Actor-Critic Model...")
        
        # Initialize model and training components
        model = ActorCritic(
            state_dim=self.configs["actor_critic"]["state_dim"],
            action_dim=self.configs["actor_critic"]["action_dim"]
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs["actor_critic"]["learning_rate"])
        
        # Training loop
        num_episodes = self.configs["actor_critic"]["num_episodes"]
        gamma = self.configs["actor_critic"]["gamma"]
        
        for episode in range(1, num_episodes + 1):
            state = self.data_generator.generate_spacecraft_state()
            done = False
            total_reward = 0
            policy_loss = 0
            value_loss = 0
            
            while not done:
                # Convert state to tensor
                state_tensor = state.to(self.device)
                
                # Get action and value from model
                action, value = model(state_tensor)
                
                # Take action in environment
                next_state, reward, done = self.data_generator.step(state_tensor, action)
                reward_tensor = torch.FloatTensor([reward]).to(self.device)
                
                # Get next value
                _, next_value = model(next_state)
                next_value = next_value.detach()
                
                # Compute losses
                value_loss = F.mse_loss(value.view(-1), reward_tensor + gamma * next_value.view(-1) * (1 - float(done)))
                policy_loss = -torch.mean(torch.log(torch.softmax(action, dim=-1)) * (reward_tensor - value.detach()))
                
                # Update model
                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update state and accumulate reward
                state = next_state
                total_reward += reward
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{num_episodes} - policy_loss: {policy_loss:.4f} - value_loss: {value_loss:.4f} - reward: {total_reward:.4f} - total_loss: {loss.item():.4f}")
        
        # Save the model
        os.makedirs("checkpoints/actor_critic", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/actor_critic/actor_critic_final.pt")
        logger.info("Saved actor-critic model")
        
        return {"final_reward": total_reward}
    
    def train_threat_detector(self) -> Dict[str, Any]:
        """Train the threat detector model."""
        logger.info("Training Threat Detector...")
        
        # Generate synthetic data
        data = self.data_generator.generate_threat_data(10000)
        features = torch.FloatTensor([d[0] for d in data])  # Extract features
        labels = torch.LongTensor([d[1] for d in data])    # Extract labels
        
        # Create TensorDataset
        dataset = TensorDataset(features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_data, batch_size=self.configs["threat_detector"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.configs["threat_detector"]["batch_size"])
        
        # Initialize model and training components
        model = ThreatDetectorNN().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs["threat_detector"]["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
        
        # Create checkpoint directory
        os.makedirs("checkpoints/threat_detector", exist_ok=True)
        
        # Setup trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            config={
                "num_epochs": self.configs["threat_detector"]["num_epochs"],
                "checkpoint_path": "checkpoints/threat_detector/threat_detector_best.pt"
            },
            device=self.device
        )
        
        # Train the model
        history = trainer.train()
        
        # Save the model
        torch.save(model.state_dict(), "checkpoints/threat_detector/threat_detector_final.pt")
        
        return history
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Train all models and return results."""
        results = {}
        
        # Train each model
        results["adversary_encoder"] = self.train_adversary_encoder()
        results["strategy_generator"] = self.train_strategy_generator()
        results["actor_critic"] = self.train_actor_critic()
        results["threat_detector"] = self.train_threat_detector()
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, Any]]):
        """Save training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.base_path / f"training_results_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save training history
        for model_name, result in results.items():
            history_path = results_dir / f"{model_name}_history.json"
            with open(history_path, "w") as f:
                json.dump(result, f, indent=4, default=str)
        
        logger.info(f"Results saved to {results_dir}")

if __name__ == "__main__":
    # Run training pipeline
    pipeline = ModelTrainingPipeline()
    results = pipeline.train_all_models()
    
    # Print summary
    print("\nTraining Summary:")
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        if "history" in result:
            history = result["history"]
            if isinstance(history, list) and len(history) > 0:
                if isinstance(history[-1], dict):
                    final_metrics = history[-1]
                    for metric, value in final_metrics.items():
                        print(f"  Final {metric}: {value:.4f}")
        print(f"  Model saved in: {result['config'].checkpoint_dir}") 