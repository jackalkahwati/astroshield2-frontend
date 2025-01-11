import os

# Database Configuration
SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Redis Configuration (optional)
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_ENABLED = os.environ.get('REDIS_ENABLED', 'false').lower() == 'true'

# Rate Limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# POL Thresholds
MAX_VELOCITY = 1000  # m/s
MAX_ACCELERATION = 100  # m/s²
MAX_RF_POWER = 1000  # watts
MIN_SAFE_DISTANCE = 1000  # meters

# Maneuver Detection
MANEUVER_THRESHOLD = 0.5  # confidence threshold
POSITION_TOLERANCE = 100  # meters

# Intent Analysis
PATTERN_WINDOW = 3600  # seconds to analyze
CONFIDENCE_THRESHOLD = 0.7
