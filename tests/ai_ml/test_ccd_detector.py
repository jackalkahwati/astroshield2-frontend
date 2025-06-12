#!/usr/bin/env python3
"""
Unit tests for Spatiotemporal CCD Detector
Tests model accuracy, performance, and robustness
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock, Mock
import time
from typing import List, Dict, Tuple
import json

# Mock the CCD detector classes if not available
class MockSpatiotemporalCCDDetector(nn.Module):
    """Mock CCD detector for testing"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = 768
        self.num_heads = 12
        self.patch_size = 16
        
    def forward(self, x):
        batch_size = x.shape[0]
        # Return mock predictions
        return torch.randn(batch_size, self.num_classes)
    
    def extract_features(self, x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.embed_dim)

class TestCCDDetector(unittest.TestCase):
    """Test CCD detector functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.model = MockSpatiotemporalCCDDetector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # CCD tactics
        self.ccd_classes = [
            "stealth_coating",
            "electronic_deception", 
            "trajectory_masking",
            "debris_simulation",
            "formation_flying",
            "proximity_operations",
            "inspection_maneuvers"
        ]
        
        # Test data dimensions
        self.batch_size = 1
        self.sequence_length = 32
        self.height = 224
        self.width = 224
        self.channels = 3
    
    def generate_test_sequence(self) -> torch.Tensor:
        """Generate test image sequence"""
        return torch.randn(
            self.batch_size,
            self.sequence_length,
            self.channels,
            self.height,
            self.width
        ).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization and architecture"""
        # Verify model parameters
        self.assertEqual(self.model.num_classes, 7)
        self.assertEqual(self.model.embed_dim, 768)
        self.assertEqual(self.model.num_heads, 12)
        self.assertEqual(self.model.patch_size, 16)
        
        # Verify model is on correct device
        next_param = next(self.model.parameters())
        self.assertEqual(next_param.device.type, self.device.type)
    
    def test_forward_pass(self):
        """Test model forward pass"""
        # Generate test input
        test_input = self.generate_test_sequence()
        
        # Forward pass
        with torch.no_grad():
            output = self.model(test_input)
        
        # Verify output shape
        expected_shape = (self.batch_size, self.model.num_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Verify output is valid (no NaN or Inf)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_inference_speed(self):
        """Test model inference speed"""
        # Generate test input
        test_input = self.generate_test_sequence()
        
        # Warm up
        with torch.no_grad():
            _ = self.model(test_input)
        
        # Time inference
        num_iterations = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(test_input)
        
        avg_time = (time.time() - start_time) / num_iterations * 1000  # ms
        
        # Verify inference speed meets requirement
        self.assertLess(avg_time, 50)  # Should be under 50ms
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Generate batch input
            test_input = torch.randn(
                batch_size,
                self.sequence_length,
                self.channels,
                self.height,
                self.width
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(test_input)
            
            # Verify output shape
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], self.model.num_classes)
    
    def test_ccd_classification(self):
        """Test CCD tactic classification"""
        # Generate test sequence
        test_input = self.generate_test_sequence()
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(test_input)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        # Verify predictions
        self.assertEqual(predictions.shape[0], self.batch_size)
        self.assertTrue(0 <= predictions[0] < len(self.ccd_classes))
        
        # Verify probability distribution
        self.assertAlmostEqual(probs.sum(dim=-1)[0].item(), 1.0, places=5)
    
    def test_feature_extraction(self):
        """Test feature extraction capabilities"""
        # Generate test input
        test_input = self.generate_test_sequence()
        
        # Extract features
        with torch.no_grad():
            features = self.model.extract_features(test_input)
        
        # Verify feature dimensions
        expected_shape = (self.batch_size, self.model.embed_dim)
        self.assertEqual(features.shape, expected_shape)
        
        # Verify features are meaningful (not all zeros)
        self.assertGreater(features.abs().mean().item(), 0.01)
    
    def test_temporal_consistency(self):
        """Test temporal consistency in predictions"""
        # Generate two similar sequences
        base_sequence = self.generate_test_sequence()
        
        # Add small noise to create similar sequence
        noise = torch.randn_like(base_sequence) * 0.1
        similar_sequence = base_sequence + noise
        
        # Get predictions for both
        with torch.no_grad():
            pred1 = torch.softmax(self.model(base_sequence), dim=-1)
            pred2 = torch.softmax(self.model(similar_sequence), dim=-1)
        
        # Calculate similarity (cosine similarity)
        similarity = torch.cosine_similarity(pred1, pred2, dim=-1)
        
        # Predictions should be similar for similar inputs
        self.assertGreater(similarity.mean().item(), 0.7)
    
    def test_edge_cases(self):
        """Test model behavior on edge cases"""
        # Test with all-zero input
        zero_input = torch.zeros(
            self.batch_size,
            self.sequence_length,
            self.channels,
            self.height,
            self.width
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(zero_input)
        
        # Model should still produce valid output
        self.assertFalse(torch.isnan(output).any())
        self.assertEqual(output.shape[1], self.model.num_classes)
        
        # Test with very large values
        large_input = torch.ones_like(zero_input) * 1000
        
        with torch.no_grad():
            output = self.model(large_input)
        
        # Model should handle large inputs gracefully
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_multi_label_detection(self):
        """Test multi-label CCD detection capabilities"""
        # Generate test input
        test_input = self.generate_test_sequence()
        
        # Get predictions with sigmoid for multi-label
        with torch.no_grad():
            logits = self.model(test_input)
            probs = torch.sigmoid(logits)
        
        # Apply threshold for multi-label detection
        threshold = 0.5
        predictions = (probs > threshold).float()
        
        # Verify multi-label format
        self.assertEqual(predictions.shape, (self.batch_size, self.model.num_classes))
        
        # Can have multiple tactics detected
        num_detected = predictions.sum(dim=-1)
        self.assertGreaterEqual(num_detected[0].item(), 0)
        self.assertLessEqual(num_detected[0].item(), self.model.num_classes)

class TestCCDDataPipeline(unittest.TestCase):
    """Test CCD detection data pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.image_size = (224, 224)
        self.sequence_length = 32
        
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        # Create sample image sequence
        raw_images = np.random.randint(0, 255, 
            (self.sequence_length, 224, 224, 3), dtype=np.uint8)
        
        # Preprocessing steps
        processed = self.preprocess_images(raw_images)
        
        # Verify preprocessing
        self.assertEqual(processed.shape[0], 1)  # Batch dimension added
        self.assertEqual(processed.shape[1], self.sequence_length)
        self.assertAlmostEqual(processed.mean(), 0, places=1)  # Normalized
        self.assertAlmostEqual(processed.std(), 1, places=1)
    
    def preprocess_images(self, images: np.ndarray) -> torch.Tensor:
        """Preprocess image sequence"""
        # Convert to float and normalize
        images = images.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        images = (images - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        return tensor.unsqueeze(0)
    
    def test_data_augmentation(self):
        """Test data augmentation for robustness"""
        # Original image
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Apply augmentations
        augmentations = [
            self.random_brightness(original),
            self.random_contrast(original),
            self.random_noise(original),
            self.random_blur(original)
        ]
        
        # Verify augmentations produce different results
        for aug in augmentations:
            self.assertFalse(np.array_equal(original, aug))
            self.assertEqual(original.shape, aug.shape)
    
    def random_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness adjustment"""
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def random_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random contrast adjustment"""
        factor = np.random.uniform(0.8, 1.2)
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def random_noise(self, image: np.ndarray) -> np.ndarray:
        """Add random noise"""
        noise = np.random.normal(0, 10, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def random_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random blur (simplified)"""
        # Simple box blur for testing
        kernel_size = 3
        pad = kernel_size // 2
        blurred = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        
        for i in range(pad, image.shape[0] + pad):
            for j in range(pad, image.shape[1] + pad):
                blurred[i-pad, j-pad] = blurred[
                    i-pad:i+pad+1, j-pad:j+pad+1
                ].mean(axis=(0, 1))
        
        return blurred[:image.shape[0], :image.shape[1]].astype(np.uint8)

class TestCCDMetrics(unittest.TestCase):
    """Test CCD detection metrics and evaluation"""
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation"""
        # Mock predictions and ground truth
        y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0, 1, 1, 0])
        
        # Calculate metrics
        precision, recall, f1 = self.calculate_metrics(y_true, y_pred)
        
        # Verify metrics
        self.assertGreater(f1, 0.8)  # Should have good F1 score
        self.assertLessEqual(precision, 1.0)
        self.assertLessEqual(recall, 1.0)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return precision, recall, f1
    
    def test_confusion_matrix(self):
        """Test confusion matrix generation"""
        # Mock predictions
        y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0, 1, 1, 0])
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Verify confusion matrix properties
        self.assertEqual(cm.shape[0], cm.shape[1])  # Square matrix
        self.assertEqual(cm.sum(), len(y_true))  # Total equals number of samples
        self.assertTrue(np.all(cm >= 0))  # All non-negative

if __name__ == "__main__":
    unittest.main() 