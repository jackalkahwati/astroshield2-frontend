# [ Imports from the original function remain: torch, nn, np, Dict, List, Tuple ]
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

# [ Import the AnomalyDetector model from its location ]
from ml.models.anomaly_detector import AnomalyDetector


def train_anomaly_detector(
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    model_params: Dict = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[AnomalyDetector, Dict[str, List[float]]]:
    \"\"\"Train the anomaly detection model\"\"\"
    # Initialize model
    if model_params is None:
        # Determine input dimensions from data if not provided
        input_dims_from_data = {}
        if train_data:
            # Need to handle different data structures (e.g., temporal)
            # This is a simplification - assumes all data has a last dimension size
            input_dims_from_data = {
                k: v.shape[-1] for k, v in train_data.items()
            }
        else: # Cannot infer dims if no data
             raise ValueError(\"Cannot infer input_dims without train_data or model_params\")

        model_params = {
            \'input_dims\': input_dims_from_data,
            \'hidden_size\': 128,
            \'latent_dim\': 64,
            \'num_flows\': 3
        }

    model = AnomalyDetector(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss functions
    mse_loss = nn.MSELoss()
    # Using a simple L2 penalty on latent space or skipping KL loss if flows are simplified

    # Training history
    history = {
        \'train_loss\': [],
        \'val_loss\': [],
        \'train_scores\': [],
        \'val_scores\': []
    }

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_scores = []

        # --- Training Batch Loop --- #
        if not train_data:
            print(\"Warning: No training data provided.\")
            break
        num_samples = len(next(iter(train_data.values())))

        for i in range(0, num_samples, batch_size):
            # Prepare batch
            batch = {
                k: v[i:i+batch_size].to(model.device) # Ensure batch is on correct device
                for k, v in train_data.items()
            }

            # Forward pass
            outputs = model(batch)

            # Calculate losses
            recon_loss = sum(
                mse_loss(outputs[\'reconstructions\'][k], batch[k])
                for k in batch.keys() if k in outputs[\'reconstructions\']
            )

            # Simplified loss: Reconstruction only
            loss = recon_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Calculate average anomaly scores
            anomaly_scores = outputs.get(\'anomaly_scores\', {})
            if anomaly_scores:
                 avg_score = torch.mean(torch.cat([
                     score.view(-1) for score in anomaly_scores.values()
                 ])).item()
                 train_scores.append(avg_score)
            else:
                 train_scores.append(0.0)

        # --- Validation --- #
        model.eval()
        val_losses = []
        val_scores = []

        if not val_data:
             print(\"Warning: No validation data provided. Skipping validation.\")
        else:
             num_val_samples = len(next(iter(val_data.values())))
             with torch.no_grad():
                 for i in range(0, num_val_samples, batch_size):
                     # Prepare batch
                     batch = {
                         k: v[i:i+batch_size].to(model.device) # Ensure batch is on correct device
                         for k, v in val_data.items()
                     }

                     # Forward pass
                     outputs = model(batch)

                     # Calculate losses
                     recon_loss = sum(
                         mse_loss(outputs[\'reconstructions\'][k], batch[k])
                         for k in batch.keys() if k in outputs[\'reconstructions\']
                     )

                     loss = recon_loss
                     val_losses.append(loss.item())

                     # Calculate average anomaly scores
                     anomaly_scores = outputs.get(\'anomaly_scores\', {})
                     if anomaly_scores:
                          avg_score = torch.mean(torch.cat([
                              score.view(-1) for score in anomaly_scores.values()
                          ])).item()
                          val_scores.append(avg_score)
                     else:
                          val_scores.append(0.0)

        # Update history
        history[\'train_loss\'].append(np.mean(train_losses) if train_losses else 0)
        history[\'val_loss\'].append(np.mean(val_losses) if val_losses else 0)
        history[\'train_scores\'].append(np.mean(train_scores) if train_scores else 0)
        history[\'val_scores\'].append(np.mean(val_scores) if val_scores else 0)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f\"Epoch {epoch+1}/{num_epochs}\")
            print(f\"Train Loss: {history[\'train_loss\'][-1]:.4f}\")
            print(f\"Val Loss: {history[\'val_loss\'][-1]:.4f}\")
            print(f\"Train Scores: {history[\'train_scores\'][-1]:.4f}\")
            print(f\"Val Scores: {history[\'val_scores\'][-1]:.4f}\")

    return model, history 