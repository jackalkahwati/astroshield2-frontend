import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert dictionary data to feature array"""
        features = []
        if 'position' in data:
            features.extend([
                data['position'].get('x', 0),
                data['position'].get('y', 0),
                data['position'].get('z', 0)
            ])
        if 'velocity' in data:
            features.extend([
                data['velocity'].get('vx', 0),
                data['velocity'].get('vy', 0),
                data['velocity'].get('vz', 0)
            ])
        return np.array(features).reshape(1, -1)
    
    def fit(self, data: List[Dict[str, Any]]):
        """Train the anomaly detection model"""
        X = np.vstack([self.prepare_features(d) for d in data])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
    
    def predict(self, data: Dict[str, Any]) -> float:
        """Predict anomaly score for new data"""
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        score = self.model.score_samples(X_scaled)[0]
        return 1 / (1 + np.exp(-score))  # Convert to probability
    
    def save_model(self, path: str):
        """Save model to disk"""
        dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        saved = load(path)
        self.model = saved['model']
        self.scaler = saved['scaler']

class BehaviorClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert dictionary data to feature array"""
        features = []
        # Extract numerical features
        if 'power' in data:
            features.append(float(data['power']))
        if 'duration' in data:
            features.append(float(data['duration']))
        if 'confidence' in data:
            features.append(float(data['confidence']))
        
        # Add derived features
        if 'position' in data and 'velocity' in data:
            pos = data['position']
            vel = data['velocity']
            speed = np.sqrt(vel['vx']**2 + vel['vy']**2 + vel['vz']**2)
            features.append(speed)
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, data: List[Dict[str, Any]], labels: List[str]):
        """Train the behavior classifier"""
        X = np.vstack([self.prepare_features(d) for d in data])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Predict behavior probabilities"""
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]
        return dict(zip(self.model.classes_, probs))
    
    def save_model(self, path: str):
        """Save model to disk"""
        dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        saved = load(path)
        self.model = saved['model']
        self.scaler = saved['scaler']

class EclipseAnalyzer:
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.behavior_classifier = BehaviorClassifier()
    
    def analyze_eclipse_behavior(self, 
                               tracking_data: Dict[str, Any],
                               eclipse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze object behavior during eclipse"""
        # Combine tracking and eclipse data
        combined_data = {**tracking_data, **eclipse_data}
        
        # Get anomaly score
        anomaly_score = self.anomaly_detector.predict(combined_data)
        
        # Get behavior classification
        behavior_probs = self.behavior_classifier.predict(combined_data)
        
        return {
            'anomaly_score': float(anomaly_score),
            'behavior_probabilities': behavior_probs,
            'confidence': min(anomaly_score, max(behavior_probs.values()))
        }

class StimulationAnalyzer:
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.behavior_classifier = BehaviorClassifier()
    
    def analyze_stimulation_response(self, 
                                   event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze object response to stimulation"""
        # Get anomaly score
        anomaly_score = self.anomaly_detector.predict(event_data)
        
        # Get behavior classification
        behavior_probs = self.behavior_classifier.predict(event_data)
        
        # Calculate response characteristics
        response_metrics = self._calculate_response_metrics(event_data)
        
        return {
            'anomaly_score': float(anomaly_score),
            'behavior_probabilities': behavior_probs,
            'response_metrics': response_metrics,
            'confidence': min(anomaly_score, max(behavior_probs.values()))
        }
    
    def _calculate_response_metrics(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate response metrics from event data"""
        metrics = {}
        if 'response_characteristics' in event_data:
            resp = event_data['response_characteristics']
            metrics['response_time'] = resp.get('duration', 0)
            metrics['response_power'] = resp.get('power', 0)
            if 'power' in resp and 'duration' in resp:
                metrics['energy'] = resp['power'] * resp['duration']
        return metrics
