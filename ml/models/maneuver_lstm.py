import torch
import torch.nn as nn

class ManeuverLSTMAutoencoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, sequence_length=60):
        """
        LSTM Autoencoder for maneuver detection
        Args:
            input_size: Number of features (position[3] + velocity[3])
            hidden_size: Size of LSTM hidden layer
            num_layers: Number of LSTM layers
            sequence_length: Number of time steps (60 days)
        """
        super(ManeuverLSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Intermediate layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Anomaly detection thresholds
        self.reconstruction_threshold = 0.1
        self.maneuver_confidence_threshold = 0.95

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        return hidden[-1]

    def decode(self, x):
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        output, _ = self.decoder(x)
        return output

    def forward(self, x):
        # Encode
        encoded = self.encode(x)
        
        # Dense layers
        x = self.fc1(encoded)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        
        # Decode
        decoded = self.decode(x)
        return decoded

    def detect_maneuver(self, trajectory_sequence, threshold=None):
        """
        Detect maneuvers in trajectory sequence
        Args:
            trajectory_sequence: Tensor of shape (batch_size, sequence_length, input_size)
            threshold: Optional custom threshold
        Returns:
            maneuver_detected: Boolean
            confidence: Float between 0 and 1
        """
        if threshold is None:
            threshold = self.reconstruction_threshold

        self.eval()
        with torch.no_grad():
            # Get reconstruction
            reconstructed = self.forward(trajectory_sequence)
            
            # Calculate reconstruction error
            error = torch.mean((trajectory_sequence - reconstructed) ** 2, dim=(1, 2))
            
            # Calculate confidence based on error distribution
            confidence = 1.0 - torch.sigmoid(error)
            
            # Detect maneuver based on threshold
            maneuver_detected = confidence > self.maneuver_confidence_threshold

        return maneuver_detected, confidence.item()

    def analyze_pol(self, trajectory_history, current_sequence):
        """
        Analyze pattern of life
        Args:
            trajectory_history: Historical trajectory sequences
            current_sequence: Current trajectory sequence
        Returns:
            pol_violation: Boolean
            confidence: Float between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            # Get historical reconstruction errors
            historical_errors = []
            for sequence in trajectory_history:
                reconstructed = self.forward(sequence)
                error = torch.mean((sequence - reconstructed) ** 2)
                historical_errors.append(error.item())
            
            # Get current reconstruction error
            current_reconstructed = self.forward(current_sequence)
            current_error = torch.mean((current_sequence - current_reconstructed) ** 2)
            
            # Calculate Z-score of current error
            mean_error = torch.tensor(historical_errors).mean()
            std_error = torch.tensor(historical_errors).std()
            z_score = (current_error - mean_error) / std_error
            
            # Calculate confidence
            confidence = torch.sigmoid(z_score).item()
            pol_violation = confidence > self.maneuver_confidence_threshold

        return pol_violation, confidence

    def predict_gaps(self, pre_gap_sequence):
        """
        Predict trajectory during sensor gaps
        Args:
            pre_gap_sequence: Sequence before the gap
        Returns:
            predicted_sequence: Predicted trajectory during gap
            confidence: Prediction confidence
        """
        self.eval()
        with torch.no_grad():
            # Encode pre-gap sequence
            encoded = self.encode(pre_gap_sequence)
            
            # Generate prediction
            predicted_sequence = self.decode(encoded)
            
            # Calculate prediction confidence based on historical accuracy
            confidence = torch.sigmoid(
                -torch.mean((pre_gap_sequence - self.forward(pre_gap_sequence)) ** 2)
            ).item()

        return predicted_sequence, confidence
