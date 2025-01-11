import numpy as np
import torch
from typing import Dict, List, Tuple

class SignatureDataGenerator:
    """Generator for synthetic signature data"""
    
    def __init__(
        self,
        image_size: int = 64,
        sequence_length: int = 48,
        num_channels: int = 3,  # optical, radar, IR
        num_classes: int = 5
    ):
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Define signature characteristics for each class
        self.class_signatures = {
            0: {  # Small satellite
                'size_range': (2, 5),
                'intensity_range': (0.2, 0.5),
                'pattern': 'point'
            },
            1: {  # Medium satellite
                'size_range': (5, 10),
                'intensity_range': (0.4, 0.7),
                'pattern': 'extended'
            },
            2: {  # Large satellite
                'size_range': (10, 20),
                'intensity_range': (0.6, 0.9),
                'pattern': 'complex'
            },
            3: {  # Rocket body
                'size_range': (15, 25),
                'intensity_range': (0.5, 0.8),
                'pattern': 'elongated'
            },
            4: {  # Debris
                'size_range': (1, 8),
                'intensity_range': (0.1, 0.4),
                'pattern': 'irregular'
            }
        }
    
    def generate_point_signature(
        self,
        size: float,
        intensity: float
    ) -> np.ndarray:
        """Generate point-like signature"""
        image = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        
        # Generate 2D Gaussian
        x = np.linspace(-size, size, self.image_size)
        y = np.linspace(-size, size, self.image_size)
        X, Y = np.meshgrid(x, y)
        
        # Create point signature
        image = intensity * np.exp(-(X**2 + Y**2)/(2*size**2))
        
        return image
    
    def generate_extended_signature(
        self,
        size: float,
        intensity: float
    ) -> np.ndarray:
        """Generate extended signature"""
        image = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        
        # Generate multiple Gaussians
        for _ in range(3):
            offset_x = np.random.uniform(-size/2, size/2)
            offset_y = np.random.uniform(-size/2, size/2)
            sub_size = size * np.random.uniform(0.3, 0.7)
            
            x = np.linspace(-size+offset_x, size+offset_x, self.image_size)
            y = np.linspace(-size+offset_y, size+offset_y, self.image_size)
            X, Y = np.meshgrid(x, y)
            
            image += intensity * np.exp(-(X**2 + Y**2)/(2*sub_size**2))
        
        return np.clip(image, 0, 1)
    
    def generate_complex_signature(
        self,
        size: float,
        intensity: float
    ) -> np.ndarray:
        """Generate complex signature"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Generate multiple components
        num_components = np.random.randint(4, 7)
        
        for _ in range(num_components):
            offset_x = np.random.uniform(-size, size)
            offset_y = np.random.uniform(-size, size)
            sub_size = size * np.random.uniform(0.2, 0.8)
            sub_intensity = intensity * np.random.uniform(0.5, 1.0)
            
            x = np.linspace(-size+offset_x, size+offset_x, self.image_size)
            y = np.linspace(-size+offset_y, size+offset_y, self.image_size)
            X, Y = np.meshgrid(x, y)
            
            image += sub_intensity * np.exp(-(X**2 + Y**2)/(2*sub_size**2))
        
        return np.clip(image, 0, 1)
    
    def generate_elongated_signature(
        self,
        size: float,
        intensity: float
    ) -> np.ndarray:
        """Generate elongated signature"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Generate elongated Gaussian
        angle = np.random.uniform(0, 2*np.pi)
        ratio = np.random.uniform(3, 5)
        
        x = np.linspace(-size, size, self.image_size)
        y = np.linspace(-size, size, self.image_size)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_rot = X*np.cos(angle) - Y*np.sin(angle)
        Y_rot = X*np.sin(angle) + Y*np.cos(angle)
        
        # Create elongated pattern
        image = intensity * np.exp(-(X_rot**2/(2*size**2) + Y_rot**2/(2*(size/ratio)**2)))
        
        return image
    
    def generate_irregular_signature(
        self,
        size: float,
        intensity: float
    ) -> np.ndarray:
        """Generate irregular signature"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Generate random number of fragments
        num_fragments = np.random.randint(2, 5)
        
        for _ in range(num_fragments):
            offset_x = np.random.uniform(-size, size)
            offset_y = np.random.uniform(-size, size)
            sub_size = size * np.random.uniform(0.2, 0.6)
            sub_intensity = intensity * np.random.uniform(0.3, 0.8)
            
            x = np.linspace(-size+offset_x, size+offset_x, self.image_size)
            y = np.linspace(-size+offset_y, size+offset_y, self.image_size)
            X, Y = np.meshgrid(x, y)
            
            # Add random shape variations
            shape_var = np.random.uniform(0.5, 1.5, (2,))
            image += sub_intensity * np.exp(
                -(X**2/(2*(sub_size*shape_var[0])**2) + 
                  Y**2/(2*(sub_size*shape_var[1])**2))
            )
        
        return np.clip(image, 0, 1)
    
    def generate_signature_sequence(
        self,
        class_id: int
    ) -> np.ndarray:
        """Generate a sequence of signature images"""
        signature_params = self.class_signatures[class_id]
        size = np.random.uniform(*signature_params['size_range'])
        intensity = np.random.uniform(*signature_params['intensity_range'])
        
        # Generate sequence
        sequence = np.zeros((self.sequence_length, self.num_channels, 
                           self.image_size, self.image_size))
        
        for t in range(self.sequence_length):
            # Add time variation
            time_size = size * (1 + 0.1*np.sin(2*np.pi*t/self.sequence_length))
            time_intensity = intensity * (1 + 0.1*np.cos(2*np.pi*t/self.sequence_length))
            
            # Generate signature based on pattern type
            if signature_params['pattern'] == 'point':
                base_signature = self.generate_point_signature(time_size, time_intensity)
            elif signature_params['pattern'] == 'extended':
                base_signature = self.generate_extended_signature(time_size, time_intensity)
            elif signature_params['pattern'] == 'complex':
                base_signature = self.generate_complex_signature(time_size, time_intensity)
            elif signature_params['pattern'] == 'elongated':
                base_signature = self.generate_elongated_signature(time_size, time_intensity)
            else:  # irregular
                base_signature = self.generate_irregular_signature(time_size, time_intensity)
            
            # Generate different channels
            for c in range(self.num_channels):
                # Add channel-specific variations
                channel_signature = base_signature * np.random.uniform(0.8, 1.2)
                sequence[t, c] = channel_signature
        
        return sequence
    
    def generate_data(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic signature data"""
        # Initialize arrays
        images = np.zeros((num_samples, self.sequence_length, self.num_channels,
                          self.image_size, self.image_size))
        labels = np.zeros(num_samples, dtype=np.int64)
        
        # Generate samples
        for i in range(num_samples):
            # Select random class
            class_id = np.random.randint(self.num_classes)
            
            # Generate signature sequence
            sequence = self.generate_signature_sequence(class_id)
            
            # Store data
            images[i] = sequence
            labels[i] = class_id
        
        # Convert to tensors
        images_tensor = torch.tensor(images, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'images': images_tensor,
            'labels': labels_tensor
        } 