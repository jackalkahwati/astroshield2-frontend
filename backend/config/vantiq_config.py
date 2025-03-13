"""
Vantiq Configuration Module

This module contains configuration settings for Vantiq integration.
"""

import os
from typing import Dict, Any

# Vantiq connection settings
VANTIQ_API_URL = os.environ.get("VANTIQ_API_URL", "https://dev.vantiq.com/api/v1")
VANTIQ_API_TOKEN = os.environ.get("VANTIQ_API_TOKEN", "")
VANTIQ_NAMESPACE = os.environ.get("VANTIQ_NAMESPACE", "asttroshield")

# Vantiq authentication settings
VANTIQ_AUTH = {
    "username": os.environ.get("VANTIQ_USERNAME", ""),
    "password": os.environ.get("VANTIQ_PASSWORD", ""),
    "client_id": os.environ.get("VANTIQ_CLIENT_ID", ""),
    "client_secret": os.environ.get("VANTIQ_CLIENT_SECRET", ""),
}

# Vantiq topic mappings
VANTIQ_TOPICS = {
    "trajectory_updates": os.environ.get("VANTIQ_TRAJECTORY_TOPIC", "TRAJECTORY_UPDATES"),
    "threat_detections": os.environ.get("VANTIQ_THREAT_TOPIC", "THREAT_DETECTIONS"),
    "commands": os.environ.get("VANTIQ_COMMAND_TOPIC", "COMMANDS"),
}

# Vantiq webhook configuration
VANTIQ_WEBHOOK_CONFIG = {
    "enabled": os.environ.get("VANTIQ_WEBHOOK_ENABLED", "true").lower() == "true",
    "secret": os.environ.get("VANTIQ_WEBHOOK_SECRET", ""),
    "endpoint": os.environ.get("VANTIQ_WEBHOOK_ENDPOINT", "/api/vantiq/webhook"),
}

# Vantiq retry configuration
VANTIQ_RETRY_CONFIG = {
    "max_retries": int(os.environ.get("VANTIQ_MAX_RETRIES", "3")),
    "retry_delay": int(os.environ.get("VANTIQ_RETRY_DELAY", "1000")),  # in milliseconds
    "timeout": int(os.environ.get("VANTIQ_TIMEOUT", "5000")),  # in milliseconds
}

def get_vantiq_config() -> Dict[str, Any]:
    """
    Get the complete Vantiq configuration
    """
    return {
        "api_url": VANTIQ_API_URL,
        "api_token": VANTIQ_API_TOKEN,
        "namespace": VANTIQ_NAMESPACE,
        "auth": VANTIQ_AUTH,
        "topics": VANTIQ_TOPICS,
        "webhook": VANTIQ_WEBHOOK_CONFIG,
        "retry": VANTIQ_RETRY_CONFIG,
    } 