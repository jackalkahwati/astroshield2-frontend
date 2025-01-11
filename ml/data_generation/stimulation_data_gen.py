import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class StimulationDataGenerator:
    def __init__(self,
                 sequence_length: int = 100,
                 input_dim: int = 128):
        """Initialize the stimulation data generator.
        
        Args:
            sequence_length: Length of input sequences
            input_dim: Dimension of input features
        """
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
    def generate_base_sequence(self) -> np.ndarray:
        """Generate a base sequence with random patterns.
        
        Returns:
            Base sequence array
        """
        # Generate random sequence
        sequence = np.random.randn(self.sequence_length, self.input_dim)
        
        # Add some temporal patterns
        for i in range(1, self.sequence_length):
            # Add autoregressive component
            sequence[i] += 0.3 * sequence[i-1]
            
            # Add some periodic patterns
            sequence[i] += 0.2 * np.sin(2 * np.pi * i / 20)
            
        return sequence
    
    def compute_responses(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute immediate and delayed responses for a sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Tuple of (immediate_response, delayed_response)
        """
        # Compute immediate response (based on current state)
        # Now computing 3D response (x, y, z)
        immediate_response = np.zeros((len(sequence), 3))
        for i in range(len(sequence)):
            # X component based on first third of features
            immediate_response[i, 0] = np.mean(sequence[i, :self.input_dim//3])
            # Y component based on middle third
            immediate_response[i, 1] = np.mean(sequence[i, self.input_dim//3:2*self.input_dim//3])
            # Z component based on last third
            immediate_response[i, 2] = np.mean(sequence[i, 2*self.input_dim//3:])
        
        # Normalize to [-1, 1]
        immediate_response = np.tanh(immediate_response)
        
        # Compute delayed response (based on sequence history)
        delayed_response = np.zeros_like(immediate_response)
        window_size = 10
        for i in range(window_size, len(sequence)):
            # Consider past window_size steps
            window = sequence[i-window_size:i]
            # X component
            delayed_response[i, 0] = np.mean(window[:, :self.input_dim//3])
            # Y component
            delayed_response[i, 1] = np.mean(window[:, self.input_dim//3:2*self.input_dim//3])
            # Z component
            delayed_response[i, 2] = np.mean(window[:, 2*self.input_dim//3:])
        
        # Normalize to [-1, 1]
        delayed_response = np.tanh(delayed_response)
        
        return immediate_response, delayed_response
    
    def compute_stability(self, sequence: np.ndarray, immediate_response: np.ndarray, delayed_response: np.ndarray) -> np.ndarray:
        """Compute stability scores for the sequence.
        
        Args:
            sequence: Input sequence
            immediate_response: Immediate response values
            delayed_response: Delayed response values
            
        Returns:
            Stability scores
        """
        # Compute stability based on:
        # 1. Variance in the sequence
        # 2. Difference between immediate and delayed responses
        # 3. Rate of change in responses
        
        # Sequence variance (lower is more stable)
        seq_variance = np.var(sequence, axis=1)
        
        # Response consistency (lower difference is more stable)
        response_diff = np.mean(np.abs(immediate_response - delayed_response), axis=1)
        
        # Rate of change (lower is more stable)
        immediate_change = np.mean(np.abs(np.diff(immediate_response, axis=0)), axis=1)
        immediate_change = np.append(immediate_change, immediate_change[-1])
        delayed_change = np.mean(np.abs(np.diff(delayed_response, axis=0)), axis=1)
        delayed_change = np.append(delayed_change, delayed_change[-1])
        
        # Combine factors
        stability = 1.0 - (
            0.4 * seq_variance / seq_variance.max() +
            0.3 * response_diff / response_diff.max() +
            0.15 * immediate_change / immediate_change.max() +
            0.15 * delayed_change / delayed_change.max()
        )
        
        # Ensure values are in [0, 1]
        stability = np.clip(stability, 0, 1)
        
        return stability
    
    def generate_stimulation_data(self, num_samples: int) -> Dict[str, np.ndarray]:
        """Generate training data for the stimulation network.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing sequences and their labels
        """
        # Initialize arrays
        sequences = np.zeros((num_samples, self.sequence_length, self.input_dim))
        immediate_responses = np.zeros((num_samples, 3))  # 3D response
        delayed_responses = np.zeros((num_samples, 3))    # 3D response
        stability_scores = np.zeros((num_samples, 1))
        
        # Generate data
        for i in range(num_samples):
            # Generate sequence
            sequence = self.generate_base_sequence()
            
            # Compute responses
            immediate, delayed = self.compute_responses(sequence)
            
            # Compute stability
            stability = self.compute_stability(sequence, immediate, delayed)
            
            # Store results
            sequences[i] = sequence
            immediate_responses[i] = immediate[-1]  # Final 3D response
            delayed_responses[i] = delayed[-1]     # Final 3D response
            stability_scores[i] = stability[-1]    # Final stability
            
            # Log progress
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        return {
            "sequences": sequences,
            "immediate_responses": immediate_responses,
            "delayed_responses": delayed_responses,
            "stability_scores": stability_scores
        }

if __name__ == "__main__":
    # Test data generation
    logging.basicConfig(level=logging.INFO)
    generator = StimulationDataGenerator()
    data = generator.generate_stimulation_data(100)
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Sequences shape: {data['sequences'].shape}")
    print(f"Immediate responses shape: {data['immediate_responses'].shape}")
    print(f"Delayed responses shape: {data['delayed_responses'].shape}")
    print(f"Stability scores shape: {data['stability_scores'].shape}")
    
    print("\nValue ranges:")
    print(f"Immediate responses: [{data['immediate_responses'].min():.3f}, {data['immediate_responses'].max():.3f}]")
    print(f"Delayed responses: [{data['delayed_responses'].min():.3f}, {data['delayed_responses'].max():.3f}]")
    print(f"Stability scores: [{data['stability_scores'].min():.3f}, {data['stability_scores'].max():.3f}]") 