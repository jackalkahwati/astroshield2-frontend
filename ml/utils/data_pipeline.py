import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@dataclass
class TelemetryDataConfig:
    feature_columns: List[str]
    target_columns: List[str]
    sequence_length: int = 100
    batch_size: int = 32
    train_split: float = 0.8
    validation_split: float = 0.1
    random_seed: int = 42

class TelemetryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        config: TelemetryDataConfig,
        scaler: Optional[StandardScaler] = None,
        is_training: bool = True
    ):
        self.data = data
        self.config = config
        self.is_training = is_training
        
        # Initialize or fit scaler
        if scaler is None and is_training:
            self.scaler = StandardScaler()
            self.scaler.fit(data[config.feature_columns])
        else:
            self.scaler = scaler
        
        # Scale features
        self.scaled_features = self.scaler.transform(data[config.feature_columns])
        self.targets = data[config.target_columns].values
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create sequences for time series processing."""
        sequences = []
        for i in range(len(self.scaled_features) - self.config.sequence_length + 1):
            seq_x = self.scaled_features[i:i + self.config.sequence_length]
            seq_y = self.targets[i + self.config.sequence_length - 1]
            sequences.append((seq_x, seq_y))
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.sequences[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class TelemetryDataPipeline:
    def __init__(self, config: TelemetryDataConfig):
        self.config = config
        self.scaler = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data splits and create data loaders."""
        # Shuffle data
        data = data.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)
        
        # Calculate split indices
        n = len(data)
        train_idx = int(n * self.config.train_split)
        val_idx = int(n * (self.config.train_split + self.config.validation_split))
        
        # Create splits
        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]
        
        # Create datasets
        self.train_dataset = TelemetryDataset(
            train_data,
            self.config,
            scaler=None,
            is_training=True
        )
        self.scaler = self.train_dataset.scaler
        
        self.val_dataset = TelemetryDataset(
            val_data,
            self.config,
            scaler=self.scaler,
            is_training=False
        )
        
        self.test_dataset = TelemetryDataset(
            test_data,
            self.config,
            scaler=self.scaler,
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def preprocess_telemetry(
        self,
        telemetry_data: pd.DataFrame,
        additional_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Preprocess telemetry data with feature engineering."""
        df = telemetry_data.copy()
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Add timestamp-based features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            
        # Add rolling statistics
        for col in self.config.feature_columns:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=10).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=10).std()
        
        # Add additional features if specified
        if additional_features:
            for feature in additional_features:
                if feature not in df.columns:
                    print(f"Warning: Additional feature {feature} not found in data")
        
        # Handle any remaining missing values from rolling calculations
        df = df.fillna(method='bfill')
        
        return df
    
    def normalize_features(
        self,
        features: np.ndarray,
        inverse: bool = False
    ) -> np.ndarray:
        """Normalize or denormalize features using the fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Run prepare_data first.")
        
        if inverse:
            return self.scaler.inverse_transform(features)
        return self.scaler.transform(features)
    
    def save_pipeline(self, path: str):
        """Save pipeline configuration and scaler."""
        import joblib
        pipeline_state = {
            'config': self.config,
            'scaler': self.scaler
        }
        joblib.dump(pipeline_state, path)
    
    def load_pipeline(self, path: str):
        """Load pipeline configuration and scaler."""
        import joblib
        pipeline_state = joblib.load(path)
        self.config = pipeline_state['config']
        self.scaler = pipeline_state['scaler'] 