"""
Configuration for UDL data collection and training example generation
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration dictionary
config = {
    # API configuration
    "udl_base_url": os.getenv("UDL_BASE_URL", "https://api.udl.io"),
    "udl_api_version": os.getenv("UDL_API_VERSION", "v1"),
    
    # Data collection
    "topics": [
        "aircraft",       # Confirmed from UI
        "conjunction", 
        "maneuver", 
        "statevector",
        "elset",
        "eoobservation",
        "radarobservation",
        "rfobservation",
        "launch",
        "onorbit"
    ],
    
    # Topic-specific configuration (max records to collect)
    # Reduced limits to save disk space
    "topic_limits": {
        "aircraft": 10,
        "conjunction": 10,
        "maneuver": 20,
        "statevector": 50,
        "elset": 10,
        "eoobservation": 30,
        "radarobservation": 30,
        "rfobservation": 20,
        "launch": 5,
        "onorbit": 50
    },
    
    # Output configuration
    "output_dir": os.getenv("OUTPUT_DIR", "./training_data"),
    "raw_data_dir": "raw_data",
    "generate_examples": True,
}