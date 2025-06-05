"""
Intent Classification Module for AstroShield

This module implements a transformer-based intent classifier that analyzes maneuver 
sequences and patterns to predict satellite intent. It integrates with the Kafka 
event processing pipeline and provides structured output for downstream analysis.

Key capabilities:
- Sequence modeling of maneuver patterns
- Multi-factor intent classification
- Uncertainty quantification
- Real-time processing with batch optimization
- Integration with feedback systems for continuous learning
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import hashlib

from .models import (
    ManeuverEvent, IntentClassificationResult, IntentClass, 
    ManeuverType, ModelConfig, AIAnalysisMessage
)
from app.common.logging import logger

try:
    import torch
    import torch.nn as nn
    from transformers import GPT2Model, GPT2Config
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, using mock implementation")
    TORCH_AVAILABLE = False


class ManeuverSequenceEncoder:
    """Encodes maneuver sequences for transformer input."""
    
    def __init__(self, max_sequence_length: int = 128):
        self.max_sequence_length = max_sequence_length
        self.feature_dim = 12  # orbital elements + delta_v + burn_duration + time_diff
        
    def encode_maneuver(self, maneuver: ManeuverEvent, reference_time: datetime) -> np.ndarray:
        """Encode a single maneuver into feature vector."""
        time_diff = (maneuver.timestamp - reference_time).total_seconds() / 3600.0  # hours
        
        # Extract orbital element changes
        before = maneuver.orbital_elements_before
        after = maneuver.orbital_elements_after
        
        features = [
            after.get('semi_major_axis', 0) - before.get('semi_major_axis', 0),
            after.get('eccentricity', 0) - before.get('eccentricity', 0),
            after.get('inclination', 0) - before.get('inclination', 0),
            after.get('raan', 0) - before.get('raan', 0),
            after.get('argument_of_perigee', 0) - before.get('argument_of_perigee', 0),
            after.get('mean_anomaly', 0) - before.get('mean_anomaly', 0),
            maneuver.delta_v,
            maneuver.burn_duration or 0.0,
            time_diff,
            self._encode_maneuver_type(maneuver.maneuver_type),
            maneuver.confidence,
            1.0  # maneuver indicator
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _encode_maneuver_type(self, maneuver_type: ManeuverType) -> float:
        """Encode maneuver type as numerical value."""
        encoding = {
            ManeuverType.PROGRADE: 1.0,
            ManeuverType.RETROGRADE: -1.0,
            ManeuverType.NORMAL: 0.5,
            ManeuverType.ANTI_NORMAL: -0.5,
            ManeuverType.RADIAL: 0.25,
            ManeuverType.ANTI_RADIAL: -0.25,
            ManeuverType.COMBINED: 0.0,
            ManeuverType.UNKNOWN: 0.0
        }
        return encoding.get(maneuver_type, 0.0)
    
    def encode_sequence(self, maneuvers: List[ManeuverEvent]) -> np.ndarray:
        """Encode sequence of maneuvers."""
        if not maneuvers:
            return np.zeros((self.max_sequence_length, self.feature_dim), dtype=np.float32)
        
        # Sort by timestamp
        sorted_maneuvers = sorted(maneuvers, key=lambda x: x.timestamp)
        reference_time = sorted_maneuvers[0].timestamp
        
        # Encode each maneuver
        encoded = []
        for maneuver in sorted_maneuvers[:self.max_sequence_length]:
            encoded.append(self.encode_maneuver(maneuver, reference_time))
        
        # Pad sequence if necessary
        while len(encoded) < self.max_sequence_length:
            encoded.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        return np.stack(encoded)


class IntentClassifierModel:
    """Transformer-based intent classification model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.encoder = ManeuverSequenceEncoder()
        self.intent_classes = list(IntentClass)
        self.num_classes = len(self.intent_classes)
        
        if TORCH_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Using mock model implementation")
    
    def _initialize_model(self):
        """Initialize the transformer model."""
        try:
            # Configure transformer
            transformer_config = GPT2Config(
                vocab_size=1,  # Not used for regression
                n_positions=self.encoder.max_sequence_length,
                n_embd=self.encoder.feature_dim * 4,  # Embedding dimension
                n_layer=6,  # Number of transformer layers
                n_head=8,   # Number of attention heads
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
            
            # Custom classification head
            class IntentClassificationHead(nn.Module):
                def __init__(self, config, num_classes):
                    super().__init__()
                    self.transformer = GPT2Model(config)
                    self.classifier = nn.Sequential(
                        nn.Linear(config.n_embd, config.n_embd // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(config.n_embd // 2, num_classes)
                    )
                    self.input_projection = nn.Linear(self.encoder.feature_dim, config.n_embd)
                
                def forward(self, x):
                    # Project input features to embedding dimension
                    x = self.input_projection(x)
                    
                    # Pass through transformer
                    outputs = self.transformer(inputs_embeds=x)
                    
                    # Use last hidden state for classification
                    last_hidden = outputs.last_hidden_state[:, -1, :]
                    logits = self.classifier(last_hidden)
                    
                    return torch.softmax(logits, dim=-1)
            
            self.model = IntentClassificationHead(transformer_config, self.num_classes)
            self.model.eval()
            
            logger.info(f"Initialized intent classifier model: {self.config.model_name} v{self.config.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.model = None
    
    def predict(self, maneuver_sequence: List[ManeuverEvent]) -> Dict[str, Any]:
        """Predict intent from maneuver sequence."""
        if not self.model and TORCH_AVAILABLE:
            logger.warning("Model not initialized, using heuristic fallback")
            return self._heuristic_classification(maneuver_sequence)
        
        try:
            # Encode sequence
            encoded_sequence = self.encoder.encode_sequence(maneuver_sequence)
            
            if TORCH_AVAILABLE and self.model:
                # Convert to tensor and add batch dimension
                input_tensor = torch.FloatTensor(encoded_sequence).unsqueeze(0)
                
                with torch.no_grad():
                    probabilities = self.model(input_tensor).numpy()[0]
                
                # Get top prediction
                predicted_idx = np.argmax(probabilities)
                confidence = float(probabilities[predicted_idx])
                predicted_class = self.intent_classes[predicted_idx]
                
                return {
                    'intent_class': predicted_class,
                    'confidence': confidence,
                    'class_probabilities': {
                        intent_class.value: float(prob) 
                        for intent_class, prob in zip(self.intent_classes, probabilities)
                    }
                }
            else:
                return self._heuristic_classification(maneuver_sequence)
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._heuristic_classification(maneuver_sequence)
    
    def _heuristic_classification(self, maneuver_sequence: List[ManeuverEvent]) -> Dict[str, Any]:
        """Fallback heuristic classification when ML model unavailable."""
        if not maneuver_sequence:
            return {
                'intent_class': IntentClass.UNKNOWN,
                'confidence': 0.1,
                'class_probabilities': {intent.value: 0.1 for intent in self.intent_classes}
            }
        
        # Simple heuristics based on maneuver characteristics
        latest_maneuver = maneuver_sequence[-1]
        delta_v = latest_maneuver.delta_v
        
        if delta_v < 0.1:
            predicted_class = IntentClass.STATION_KEEPING
            confidence = 0.7
        elif delta_v > 10.0:
            predicted_class = IntentClass.EVASION
            confidence = 0.6
        elif len(maneuver_sequence) > 3:
            predicted_class = IntentClass.INSPECTION
            confidence = 0.5
        else:
            predicted_class = IntentClass.ROUTINE_MAINTENANCE
            confidence = 0.4
        
        # Create probability distribution
        probabilities = {intent.value: 0.05 for intent in self.intent_classes}
        probabilities[predicted_class.value] = confidence
        
        return {
            'intent_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': probabilities
        }


class IntentClassifier:
    """Main intent classification service."""
    
    def __init__(self, config: Optional[ModelConfig] = None, kafka_adapter=None):
        """Initialize the intent classifier."""
        self.config = config or ModelConfig(
            model_name="intent_classifier",
            model_version="1.0.0",
            confidence_threshold=0.7
        )
        self.kafka_adapter = kafka_adapter
        self.model = IntentClassifierModel(self.config)
        self.maneuver_history: Dict[str, List[ManeuverEvent]] = {}
        self.analysis_cache: Dict[str, IntentClassificationResult] = {}
        
        logger.info(f"IntentClassifier initialized: {self.config.model_name}")
    
    async def analyze_intent(self, event: ManeuverEvent) -> IntentClassificationResult:
        """Analyze intent for a maneuver event."""
        try:
            # Add to maneuver history
            sat_id = event.primary_norad_id
            if sat_id not in self.maneuver_history:
                self.maneuver_history[sat_id] = []
            
            self.maneuver_history[sat_id].append(event)
            
            # Keep only recent maneuvers (last 30 days)
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            self.maneuver_history[sat_id] = [
                m for m in self.maneuver_history[sat_id] 
                if m.timestamp > cutoff_time
            ]
            
            # Get maneuver sequence for analysis
            sequence = self.maneuver_history[sat_id][-10:]  # Last 10 maneuvers
            
            # Predict intent
            prediction = self.model.predict(sequence)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(event, sequence, prediction)
            
            # Create result
            result = IntentClassificationResult(
                event_id=event.event_id,
                timestamp=datetime.utcnow(),
                sat_pair_id=event.sat_pair_id,
                intent_class=prediction['intent_class'],
                confidence_score=prediction['confidence'],
                maneuver_type=event.maneuver_type,
                reasoning=reasoning,
                model_version=self.config.model_version,
                source_data_lineage=event.source_data_lineage
            )
            
            # Cache result
            cache_key = self._generate_cache_key(event)
            self.analysis_cache[cache_key] = result
            
            # Publish to Kafka if adapter available
            if self.kafka_adapter:
                await self._publish_result(result)
            
            logger.info(f"Intent analysis completed: {result.intent_class} (confidence: {result.confidence_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Intent analysis failed for event {event.event_id}: {str(e)}")
            raise
    
    def _generate_reasoning(self, event: ManeuverEvent, sequence: List[ManeuverEvent], 
                          prediction: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for the classification."""
        reasoning = []
        
        # Analyze current maneuver characteristics
        if event.delta_v > 5.0:
            reasoning.append(f"High delta-V maneuver ({event.delta_v:.2f} m/s) suggests significant orbit change")
        elif event.delta_v < 0.5:
            reasoning.append(f"Low delta-V maneuver ({event.delta_v:.2f} m/s) indicates minor adjustment")
        
        # Analyze sequence patterns
        if len(sequence) > 1:
            time_gaps = []
            for i in range(1, len(sequence)):
                gap = (sequence[i].timestamp - sequence[i-1].timestamp).total_seconds() / 3600
                time_gaps.append(gap)
            
            avg_gap = np.mean(time_gaps)
            if avg_gap < 24:
                reasoning.append("Frequent maneuvers (< 24h intervals) suggest active operations")
            elif avg_gap > 720:  # 30 days
                reasoning.append("Infrequent maneuvers suggest routine maintenance")
        
        # Analyze maneuver type patterns
        maneuver_types = [m.maneuver_type for m in sequence]
        if len(set(maneuver_types)) == 1:
            reasoning.append(f"Consistent {maneuver_types[0]} maneuvers suggest systematic approach")
        elif len(set(maneuver_types)) > 3:
            reasoning.append("Varied maneuver types suggest complex operation")
        
        # Confidence-based reasoning
        confidence = prediction['confidence']
        if confidence > 0.8:
            reasoning.append("High confidence classification based on clear pattern recognition")
        elif confidence < 0.5:
            reasoning.append("Low confidence - pattern matches multiple intent categories")
        
        return reasoning
    
    def _generate_cache_key(self, event: ManeuverEvent) -> str:
        """Generate cache key for analysis result."""
        key_data = f"{event.event_id}_{event.timestamp}_{event.primary_norad_id}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _publish_result(self, result: IntentClassificationResult):
        """Publish classification result to Kafka."""
        try:
            message = AIAnalysisMessage(
                message_type="intent_classification_result",
                analysis_type="intent_classification",
                payload=result.dict(),
                correlation_id=result.event_id
            )
            
            await self.kafka_adapter.publish("astroshield.ai.intent_classification", message.dict())
            logger.debug(f"Published intent classification result for event {result.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish result: {str(e)}")
    
    async def batch_analyze(self, events: List[ManeuverEvent]) -> List[IntentClassificationResult]:
        """Analyze multiple events in batch for efficiency."""
        results = []
        
        # Process events concurrently
        tasks = [self.analyze_intent(event) for event in events]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis failed for event {events[i].event_id}: {str(result)}")
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            return []
    
    def get_analysis_history(self, sat_id: str, days: int = 7) -> List[IntentClassificationResult]:
        """Get classification history for a satellite."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        history = []
        for result in self.analysis_cache.values():
            if (result.sat_pair_id.startswith(sat_id) and 
                result.timestamp > cutoff_time):
                history.append(result)
        
        return sorted(history, key=lambda x: x.timestamp, reverse=True)
    
    def update_model(self, new_config: ModelConfig):
        """Update model configuration and reload if necessary."""
        if new_config.model_version != self.config.model_version:
            logger.info(f"Updating model from {self.config.model_version} to {new_config.model_version}")
            self.config = new_config
            self.model = IntentClassifierModel(self.config)
            self.analysis_cache.clear()  # Clear cache for new model
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get basic performance metrics."""
        total_analyses = len(self.analysis_cache)
        if total_analyses == 0:
            return {"total_analyses": 0, "average_confidence": 0.0}
        
        confidences = [result.confidence_score for result in self.analysis_cache.values()]
        avg_confidence = np.mean(confidences)
        
        # Count classifications by intent
        intent_counts = {}
        for result in self.analysis_cache.values():
            intent = result.intent_class.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "total_analyses": total_analyses,
            "average_confidence": float(avg_confidence),
            "intent_distribution": intent_counts,
            "model_version": self.config.model_version
        }


# Factory function for easy instantiation
def create_intent_classifier(config: Optional[ModelConfig] = None, 
                           kafka_adapter=None) -> IntentClassifier:
    """Create and return an IntentClassifier instance."""
    return IntentClassifier(config=config, kafka_adapter=kafka_adapter) 