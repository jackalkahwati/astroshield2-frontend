"""
Weather data integration for AstroShield.

This module provides tools for integrating weather data with AstroShield for
determining observation windows and conditions.
"""

from .weather_client import WeatherDataClient
from .analysis import analyze_weather_data

__all__ = ['WeatherDataClient', 'analyze_weather_data'] 