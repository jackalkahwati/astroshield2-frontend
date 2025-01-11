"""ML-enhanced evaluators for CCDM analysis."""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from .ccdm_evaluators import CCDMIndicator
from ml.enhanced_models import ManeuverDetectionModel, SignatureAnalysisModel, AMRAnalysisModel

logger = logging.getLogger(__name__)

class MLManeuverEvaluator:
    """Enhanced maneuver evaluator using LSTM model."""
    
    def __init__(self):
        self.model = ManeuverDetectionModel()
        
    def analyze_maneuvers(self, trajectory_data: List[Dict[str, Any]]) -> List[CCDMIndicator]:
        """Analyze trajectory data for maneuvers using ML."""
        indicators = []
        try:
            predictions = self.model.predict(trajectory_data)
            
            # Check for subtle maneuvers
            if predictions['subtle_maneuver'] > 0.3:  # Threshold for subtle maneuvers
                indicators.append(CCDMIndicator(
                    indicator_name="subtle_maneuver_detected",
                    is_detected=True,
                    confidence_level=predictions['subtle_maneuver'],
                    evidence={
                        'probabilities': predictions,
                        'trajectory_points': len(trajectory_data)
                    },
                    timestamp=datetime.utcnow()
                ))
            
            # Check for significant maneuvers
            if predictions['significant_maneuver'] > 0.5:  # Higher threshold for significant maneuvers
                indicators.append(CCDMIndicator(
                    indicator_name="significant_maneuver_detected",
                    is_detected=True,
                    confidence_level=predictions['significant_maneuver'],
                    evidence={
                        'probabilities': predictions,
                        'trajectory_points': len(trajectory_data)
                    },
                    timestamp=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.error(f"ML maneuver analysis error: {str(e)}")
            
        return indicators

class MLSignatureEvaluator:
    """Enhanced signature evaluator using CNN model."""
    
    def __init__(self):
        self.model = SignatureAnalysisModel()
        
    def analyze_signatures(self, 
                         optical_data: Dict[str, Any],
                         radar_data: Dict[str, Any]) -> List[CCDMIndicator]:
        """Analyze optical and radar signatures using ML."""
        indicators = []
        try:
            combined_data = {
                'optical': optical_data,
                'radar': radar_data
            }
            
            predictions = self.model.predict(combined_data)
            
            # Check for anomalous signatures
            if predictions['anomalous'] > 0.4:
                indicators.append(CCDMIndicator(
                    indicator_name="anomalous_signature",
                    is_detected=True,
                    confidence_level=predictions['anomalous'],
                    evidence={
                        'probabilities': predictions,
                        'signature_type': 'combined'
                    },
                    timestamp=datetime.utcnow()
                ))
            
            # Check for signature mismatch
            if predictions['mismatched'] > 0.4:
                indicators.append(CCDMIndicator(
                    indicator_name="signature_mismatch",
                    is_detected=True,
                    confidence_level=predictions['mismatched'],
                    evidence={
                        'probabilities': predictions,
                        'optical_features': list(optical_data.keys()),
                        'radar_features': list(radar_data.keys())
                    },
                    timestamp=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.error(f"ML signature analysis error: {str(e)}")
            
        return indicators

class MLAMREvaluator:
    """Enhanced AMR evaluator using Random Forest models."""
    
    def __init__(self):
        self.model = AMRAnalysisModel()
        
    def analyze_amr(self, amr_data: Dict[str, Any]) -> List[CCDMIndicator]:
        """Analyze AMR characteristics using ML."""
        indicators = []
        try:
            predictions = self.model.predict(amr_data)
            
            # Check for AMR anomalies
            if predictions['classification']['anomalous'] > 0.4:
                indicators.append(CCDMIndicator(
                    indicator_name="amr_anomaly",
                    is_detected=True,
                    confidence_level=predictions['classification']['anomalous'],
                    evidence={
                        'predicted_amr': predictions['predicted_amr'],
                        'feature_importance': predictions['feature_importance'],
                        'confidence': predictions['confidence']
                    },
                    timestamp=datetime.utcnow()
                ))
            
            # Check for significant AMR changes
            current_amr = amr_data.get('amr_value', 0)
            if abs(current_amr - predictions['predicted_amr']) > 0.2 * current_amr:  # 20% threshold
                indicators.append(CCDMIndicator(
                    indicator_name="amr_change",
                    is_detected=True,
                    confidence_level=predictions['confidence'],
                    evidence={
                        'current_amr': current_amr,
                        'predicted_amr': predictions['predicted_amr'],
                        'change_percentage': abs(current_amr - predictions['predicted_amr']) / current_amr * 100
                    },
                    timestamp=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.error(f"ML AMR analysis error: {str(e)}")
            
        return indicators
