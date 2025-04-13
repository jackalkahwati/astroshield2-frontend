from typing import Dict, Any, List

INDICATOR_SPECIFICATIONS = {
    "stability_indicators": {
        "object_stability": {
            "algorithm": "LSTM Neural Network",
            "description": "Evaluates orbital stability using time-series analysis",
            "pass_criteria": "Variance in orbital parameters < 0.1 over 30 days",
            "confidence_threshold": 0.85,
            "features": ["orbital_elements", "historical_stability"]
        },
        "stability_changes": {
            "algorithm": "Change Point Detection + Random Forest",
            "description": "Detects significant changes in stability patterns",
            "pass_criteria": "No unexplained stability changes in last 14 days",
            "confidence_threshold": 0.90,
            "features": ["historical_stability", "space_environment"]
        }
    },
    "maneuver_indicators": {
        "maneuvers_detected": {
            "algorithm": "Bi-LSTM with Attention",
            "description": "Identifies and classifies orbital maneuvers",
            "pass_criteria": "All maneuvers match declared operations",
            "confidence_threshold": 0.92,
            "features": ["trajectory_features", "maneuver_history"]
        },
        "pattern_of_life": {
            "algorithm": "Temporal Pattern Mining + Neural Network",
            "description": "Analyzes if maneuvers follow expected patterns",
            "pass_criteria": "Maneuver patterns within 2σ of historical norms",
            "confidence_threshold": 0.88,
            "features": ["maneuver_history", "rf_emissions"]
        }
    },
    "rf_indicators": {
        "rf_detection": {
            "algorithm": "Convolutional Neural Network",
            "description": "Analyzes RF emissions and patterns",
            "pass_criteria": "RF signatures match declared capabilities",
            "confidence_threshold": 0.95,
            "features": ["rf_emissions"]
        },
        "subsatellite_deployment": {
            "algorithm": "Multi-target Tracking + Random Forest",
            "description": "Detects potential subsatellite deployments",
            "pass_criteria": "No unexpected object separations",
            "confidence_threshold": 0.93,
            "features": ["trajectory_features", "rf_emissions"]
        }
    },
    "compliance_indicators": {
        "itu_fcc_compliance": {
            "algorithm": "Rule-based System + Decision Tree",
            "description": "Checks compliance with ITU/FCC regulations",
            "pass_criteria": "No violations of filed frequency/orbit parameters",
            "confidence_threshold": 0.98,
            "features": ["compliance_data", "rf_emissions"]
        },
        "analyst_consensus": {
            "algorithm": "Ensemble Voting (Multiple ML Models)",
            "description": "Evaluates agreement between analyst classifications",
            "pass_criteria": "≥80% analyst agreement on classification",
            "confidence_threshold": 0.85,
            "features": ["historical_classifications"]
        }
    },
    "signature_indicators": {
        "optical_signature": {
            "algorithm": "Deep Neural Network + Image Processing",
            "description": "Analyzes optical signature characteristics",
            "pass_criteria": "Signature matches declared physical properties",
            "confidence_threshold": 0.90,
            "features": ["signature_features"]
        },
        "radar_signature": {
            "algorithm": "3D CNN + Signal Processing",
            "description": "Analyzes radar cross-section and characteristics",
            "pass_criteria": "RCS within expected range for declared type",
            "confidence_threshold": 0.92,
            "features": ["signature_features"]
        }
    },
    "stimulation_indicators": {
        "system_response": {
            "algorithm": "Reinforcement Learning + Pattern Recognition",
            "description": "Evaluates responses to system stimulation",
            "pass_criteria": "Responses match expected behavior profile",
            "confidence_threshold": 0.94,
            "features": ["system_interactions"]
        }
    },
    "physical_indicators": {
        "area_mass_ratio": {
            "algorithm": "Physics-based ML Model",
            "description": "Analyzes area-to-mass ratio characteristics",
            "pass_criteria": "AMR consistent with declared configuration",
            "confidence_threshold": 0.91,
            "features": ["signature_features", "orbital_elements"]
        },
        "proximity_operations": {
            "algorithm": "Graph Neural Network",
            "description": "Detects and analyzes close approaches",
            "pass_criteria": "No unexplained proximity operations",
            "confidence_threshold": 0.93,
            "features": ["trajectory_features", "maneuver_history"]
        }
    },
    "tracking_indicators": {
        "tracking_anomalies": {
            "algorithm": "Anomaly Detection (Isolation Forest)",
            "description": "Identifies unusual tracking behavior",
            "pass_criteria": "No unexplained tracking gaps or anomalies",
            "confidence_threshold": 0.89,
            "features": ["orbital_features"]
        },
        "imaging_maneuvers": {
            "algorithm": "Behavioral Pattern Recognition",
            "description": "Detects potential imaging/sensing activities",
            "pass_criteria": "All sensing activities declared and authorized",
            "confidence_threshold": 0.87,
            "features": ["maneuver_history", "trajectory_features"]
        }
    },
    "launch_indicators": {
        "launch_site": {
            "algorithm": "Geospatial ML + Threat Assessment",
            "description": "Evaluates launch site characteristics",
            "pass_criteria": "Launch site verified and non-threatening",
            "confidence_threshold": 0.96,
            "features": ["launch_data"]
        },
        "un_registry": {
            "algorithm": "Document Analysis + Verification",
            "description": "Verifies UN registry status",
            "pass_criteria": "Object properly registered with UNOOSA",
            "confidence_threshold": 0.99,
            "features": ["compliance_data"]
        }
    }
}

def get_indicator_specs() -> Dict[str, Any]:
    """Return the complete indicator specifications"""
    return INDICATOR_SPECIFICATIONS

def get_indicator_categories() -> List[str]:
    """Return list of all indicator categories"""
    return list(INDICATOR_SPECIFICATIONS.keys())

def get_indicators_by_category(category: str) -> Dict[str, Any]:
    """Return indicators for a specific category"""
    return INDICATOR_SPECIFICATIONS.get(category, {}) 