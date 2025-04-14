"""Configuration loader for CCDM service with environment-specific support."""
import os
import logging
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Environment detection
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "development")
CONFIG_DIR = os.getenv("CONFIG_DIR", "./config")

def load_config() -> Dict[str, Any]:
    """
    Load configuration based on current environment.
    
    Returns:
        Dictionary with configuration values
    """
    # Default configuration
    default_config = {
        "service": {
            "name": "ccdm_service",
            "version": "1.0.0",
            "description": "CCDM Service for Astroshield"
        },
        "logging": {
            "level": "INFO",
            "structured": True,
            "log_file": "logs/ccdm_service.log",
            "rotation": {
                "max_bytes": 10485760,  # 10MB
                "backup_count": 5
            }
        },
        "database": {
            "url": "sqlite:///./astroshield.db",
            "timeout": 30,
            "pool_size": 10,
            "max_overflow": 20,
            "echo": False
        },
        "rate_limiting": {
            "enabled": True,
            "default_limit": 100,
            "endpoints": {
                "get_historical_analysis": 30,
                "analyze_conjunction": 60,
                "get_assessment": 120
            }
        },
        "caching": {
            "enabled": True,
            "default_ttl": 300,
            "endpoints": {
                "get_historical_analysis": 300,
                "get_assessment": 180
            }
        },
        "api": {
            "port": 8000,
            "host": "0.0.0.0",
            "cors": {
                "enabled": True,
                "origins": ["*"]
            },
            "authentication": {
                "enabled": True,
                "jwt_secret": "change-this-in-production",
                "token_expiry_seconds": 86400
            }
        },
        "alerting": {
            "enabled": False,
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "",
                "to_addresses": []
            },
            "slack": {
                "enabled": False,
                "webhook_url": ""
            }
        },
        "external_services": {
            "space_track": {
                "username": "",
                "password": "",
                "base_url": "https://www.space-track.org/basicspacedata/query",
                "timeout": 30
            },
            "udl": {
                "base_url": "https://udl.sda.mil/api",
                "api_key_env": "UDL_API_KEY",
                "timeout": 30
            }
        }
    }
    
    # Try to load environment-specific config
    config_path = os.path.join(CONFIG_DIR, f"{DEPLOYMENT_ENV}.yaml")
    env_config = {}
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                env_config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}")
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {str(e)}")
    
    # Merge configurations with environment config taking precedence
    def deep_merge(base, override):
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    config = deep_merge(default_config, env_config)
    
    # Override with environment variables
    _override_from_env_vars(config)
    
    logger.info(f"Configuration loaded for environment: {DEPLOYMENT_ENV}")
    
    return config

def _override_from_env_vars(config: Dict[str, Any]) -> None:
    """
    Override configuration with environment variables.
    
    Args:
        config: Configuration dictionary to update
    """
    # Database settings
    if os.getenv("DATABASE_URL"):
        config["database"]["url"] = os.getenv("DATABASE_URL")
    
    if os.getenv("DB_TIMEOUT_SECONDS"):
        config["database"]["timeout"] = int(os.getenv("DB_TIMEOUT_SECONDS"))
    
    if os.getenv("DB_POOL_SIZE"):
        config["database"]["pool_size"] = int(os.getenv("DB_POOL_SIZE"))
    
    # API settings
    if os.getenv("API_PORT"):
        config["api"]["port"] = int(os.getenv("API_PORT"))
    
    if os.getenv("API_HOST"):
        config["api"]["host"] = os.getenv("API_HOST")
    
    if os.getenv("JWT_SECRET"):
        config["api"]["authentication"]["jwt_secret"] = os.getenv("JWT_SECRET")
    
    # Rate limiting
    if os.getenv("RATE_LIMIT_ENABLED"):
        config["rate_limiting"]["enabled"] = os.getenv("RATE_LIMIT_ENABLED").lower() in ("true", "1", "yes")
    
    # Caching
    if os.getenv("CACHE_ENABLED"):
        config["caching"]["enabled"] = os.getenv("CACHE_ENABLED").lower() in ("true", "1", "yes")
    
    # Alerting
    if os.getenv("ALERTING_ENABLED"):
        config["alerting"]["enabled"] = os.getenv("ALERTING_ENABLED").lower() in ("true", "1", "yes")
    
    # External services
    if os.getenv("SPACE_TRACK_USERNAME"):
        config["external_services"]["space_track"]["username"] = os.getenv("SPACE_TRACK_USERNAME")
    
    if os.getenv("SPACE_TRACK_PASSWORD"):
        config["external_services"]["space_track"]["password"] = os.getenv("SPACE_TRACK_PASSWORD")
    
    if os.getenv("UDL_API_KEY"):
        config["external_services"]["udl"]["api_key"] = os.getenv("UDL_API_KEY")

# Global config instance
_config = None

def get_config() -> Dict[str, Any]:
    """
    Get the configuration singleton.
    
    Returns:
        Configuration dictionary
    """
    global _config
    
    if _config is None:
        _config = load_config()
        
    return _config

def reload_config() -> Dict[str, Any]:
    """
    Force reload of configuration.
    
    Returns:
        Updated configuration dictionary
    """
    global _config
    
    _config = load_config()
    return _config 