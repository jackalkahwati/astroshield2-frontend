import logging
from pathlib import Path
import json
from datetime import datetime
import sys
import traceback
import torch

# Import all trainers
from ml.training.train_launch_evaluator import LaunchEvaluatorTrainer
from ml.training.train_environmental_evaluator import EnvironmentalEvaluatorTrainer
from ml.training.train_consensus_net import ConsensusNetTrainer
from ml.training.train_compliance_net import ComplianceNetTrainer
from ml.training.train_stimulation_net import StimulationNetTrainer

# Import training functions for additional models
from ml.models.train_stability_model import train_stability_model
from ml.models.train_launch_model import train_launch_model
from ml.models.train_environmental_model import train_environmental_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def train_all_models(num_samples: int = 10000):
    """Train all models in sequence.
    
    Args:
        num_samples: Number of samples to generate for each model
    """
    try:
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"training_results_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize class-based trainers
        trainers = {
            "launch_evaluator": LaunchEvaluatorTrainer(),
            "environmental_evaluator": EnvironmentalEvaluatorTrainer(),
            "consensus_net": ConsensusNetTrainer(),
            "compliance_net": ComplianceNetTrainer(),
            "stimulation_net": StimulationNetTrainer()
        }
        
        # Define function-based trainers
        function_trainers = {
            "stability_model": train_stability_model,
            "launch_model": train_launch_model,
            "environmental_model": train_environmental_model
        }
        
        # Train class-based models
        results = {}
        total_models = len(trainers) + len(function_trainers)
        completed_models = 0
        
        logger.info("\nTraining class-based models...")
        for name, trainer in trainers.items():
            logger.info(f"\nStarting training for {name}... ({completed_models + 1}/{total_models})")
            try:
                # Move model to appropriate device
                if hasattr(trainer, 'model'):
                    trainer.model = trainer.model.to(device)
                
                # Log configuration
                logger.info(f"Configured for {trainer.config['num_epochs']} epochs")
                logger.info(f"Batch size: {trainer.config.get('batch_size', 'N/A')}")
                logger.info(f"Learning rate: {trainer.config.get('learning_rate', 'N/A')}")
                
                # Train model
                model_results = trainer.train(num_samples=num_samples)
                results[name] = model_results
                
                # Save training history
                history_path = results_dir / f"{name}_history.json"
                with open(history_path, "w") as f:
                    json.dump(model_results["history"], f, indent=4)
                
                # Log final metrics
                final_metrics = model_results["history"][-1]
                logger.info(f"{name} training completed successfully!")
                logger.info("Final metrics:")
                for metric, value in final_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
                completed_models += 1
                logger.info(f"Progress: {completed_models}/{total_models} models completed")
                    
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                logger.error("Traceback:")
                logger.error(traceback.format_exc())
                continue
        
        # Train function-based models
        logger.info("\nTraining function-based models...")
        for name, train_fn in function_trainers.items():
            logger.info(f"\nStarting training for {name}... ({completed_models + 1}/{total_models})")
            try:
                # Train model with default parameters
                model = train_fn(
                    num_epochs=50,
                    batch_size=32,
                    learning_rate=0.001,
                    train_samples=num_samples,
                    val_samples=int(num_samples * 0.2)
                )
                
                # Save model if it has a save method
                if hasattr(model, 'save'):
                    model_path = results_dir / f"{name}.pt"
                    model.save(model_path)
                
                completed_models += 1
                logger.info(f"Progress: {completed_models}/{total_models} models completed")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                logger.error("Traceback:")
                logger.error(traceback.format_exc())
                continue
        
        # Save overall results summary
        summary = {
            name: {
                "config": trainer_results["config"],
                "final_metrics": trainer_results["history"][-1],
                "training_device": str(device)
            }
            for name, trainer_results in results.items()
        }
        
        summary_path = results_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"\nTraining completed! Results saved to {results_dir}")
        logger.info(f"Successfully trained {completed_models}/{total_models} models")
        
    except Exception as e:
        logger.error(f"Fatal error in training pipeline: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    train_all_models() 