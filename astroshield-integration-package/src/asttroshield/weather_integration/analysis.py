"""
Weather data analysis for observation planning.
"""
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

def analyze_weather_data(weather_data: Dict[str, Any], object_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze weather data for observation suitability
    
    Args:
        weather_data: Weather data containing cloud cover, visibility, etc.
        object_data: Target object information
        
    Returns:
        Analysis results with observation window recommendations
    """
    # Extract key weather metrics
    location = weather_data.get("location", {})
    conditions = weather_data.get("conditions", {})
    
    cloud_cover = conditions.get("clouds", {}).get("coverage", 0.0)
    visibility_km = conditions.get("visibility", {}).get("value", 10.0)
    
    # Calculate quality factors
    cloud_factor = 1.0 - (cloud_cover / 100.0) if cloud_cover <= 100 else 0.0
    visibility_factor = min(1.0, visibility_km / 10.0)
    
    # Calculate overall quality score
    quality_score = (cloud_factor * 0.7 + visibility_factor * 0.3)
    
    # Determine quality category and recommendation
    quality_category = "EXCELLENT"
    recommendation = "GO"
    
    if quality_score < 0.7:
        quality_category = "GOOD"
    if quality_score < 0.5:
        quality_category = "FAIR"
        recommendation = "CAUTION"
    if quality_score < 0.3:
        quality_category = "POOR"
        recommendation = "NO_GO"
    
    # Create observation window recommendation
    result = {
        "location": {
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude")
        },
        "qualityScore": round(quality_score, 2),
        "qualityCategory": quality_category,
        "recommendation": recommendation,
        "observationWindow": {
            "start_time": weather_data.get("forecast_period", {}).get("start"),
            "end_time": weather_data.get("forecast_period", {}).get("end"),
            "duration_minutes": calculate_duration_minutes(
                weather_data.get("forecast_period", {}).get("start"),
                weather_data.get("forecast_period", {}).get("end")
            )
        },
        "targetObject": {
            "catalog_id": object_data.get("catalog_id"),
            "altitude_km": object_data.get("altitude_km")
        }
    }
    
    return result

def find_optimal_observation_windows(
    forecast_data: Dict[str, Any], 
    min_quality_score: float = 0.6,
    min_window_minutes: int = 30
) -> List[Dict[str, Any]]:
    """
    Find optimal observation windows from forecast data
    
    Args:
        forecast_data: Weather forecast data
        min_quality_score: Minimum quality score for a good observation window
        min_window_minutes: Minimum duration in minutes for a valid window
        
    Returns:
        List of recommended observation windows
    """
    windows = []
    
    # Get days from forecast
    days = forecast_data.get("days", [])
    location = forecast_data.get("location", {})
    
    for day in days:
        date = day.get("date")
        hourly_forecasts = day.get("hourly", [])
        
        # Get astronomy data for the day
        astronomy = day.get("astronomy", {})
        sunset = astronomy.get("sunset")
        sunrise = astronomy.get("sunrise")
        
        # Convert sunrise/sunset to 24-hour format
        sunrise_hour = _convert_12h_to_24h(sunrise) if sunrise else 6  # Default to 6 AM
        sunset_hour = _convert_12h_to_24h(sunset) if sunset else 18    # Default to 6 PM
        
        # Group consecutive good observation hours
        current_window = []
        for hour_data in hourly_forecasts:
            hour_time = hour_data.get("time")
            hour = int(hour_time.split()[1].split(":")[0]) if hour_time and " " in hour_time else 0
            
            # Check if it's nighttime (after sunset or before sunrise)
            is_night = hour >= sunset_hour or hour < sunrise_hour
            
            # Calculate quality score for this hour
            cloud_cover = hour_data.get("clouds", {}).get("coverage", 0.0)
            visibility_km = hour_data.get("visibility", {}).get("km", 10.0)
            
            cloud_factor = 1.0 - (cloud_cover / 100.0) if cloud_cover <= 100 else 0.0
            visibility_factor = min(1.0, visibility_km / 10.0)
            
            # Night observation has higher weight for cloud cover
            quality_score = (cloud_factor * (0.8 if is_night else 0.6) + 
                            visibility_factor * (0.2 if is_night else 0.4))
            
            # If quality score meets minimum threshold, add to current window
            if quality_score >= min_quality_score:
                hour_data["quality_score"] = round(quality_score, 2)
                current_window.append((hour_time, hour_data))
            else:
                # If we have a valid window, add it to the list
                if len(current_window) > 0:
                    window_duration = len(current_window) * 60  # Each entry is one hour
                    
                    if window_duration >= min_window_minutes:
                        start_time = current_window[0][0]
                        end_time = current_window[-1][0]
                        avg_quality = sum(item[1].get("quality_score", 0) for item in current_window) / len(current_window)
                        
                        windows.append({
                            "date": date,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration_minutes": window_duration,
                            "quality_score": round(avg_quality, 2),
                            "quality_category": _quality_score_to_category(avg_quality),
                            "location": {
                                "latitude": location.get("latitude"),
                                "longitude": location.get("longitude")
                            }
                        })
                        
                    # Reset window
                    current_window = []
    
    return windows

def calculate_duration_minutes(start_time: Optional[str], end_time: Optional[str]) -> int:
    """
    Calculate duration in minutes between two time strings
    
    Args:
        start_time: Start time in ISO format
        end_time: End time in ISO format
        
    Returns:
        Duration in minutes
    """
    if not start_time or not end_time:
        return 60  # Default
    
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        duration = (end - start).total_seconds() / 60
        return max(1, int(duration))
    except (ValueError, TypeError):
        return 60  # Default if parsing fails

def _convert_12h_to_24h(time_str: Optional[str]) -> int:
    """Convert 12-hour time format to hour in 24-hour format."""
    if not time_str:
        return 0
        
    try:
        if "AM" in time_str or "PM" in time_str:
            time_obj = datetime.strptime(time_str, "%I:%M %p")
            return time_obj.hour
        else:
            time_parts = time_str.split(":")
            return int(time_parts[0])
    except (ValueError, IndexError):
        return 0

def _quality_score_to_category(score: float) -> str:
    """Convert quality score to category string."""
    if score >= 0.8:
        return "EXCELLENT"
    elif score >= 0.6:
        return "GOOD"
    elif score >= 0.4:
        return "FAIR"
    else:
        return "POOR" 