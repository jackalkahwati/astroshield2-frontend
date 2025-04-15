from asttroshield.weather_integration import WeatherDataClient

# Example of analyzing observation conditions
def analyze_weather_data(weather_data, object_data):
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

def calculate_duration_minutes(start_time, end_time):
    """Calculate duration in minutes between two time strings"""
    from datetime import datetime
    
    if not start_time or not end_time:
        return 60  # Default
    
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        duration = (end - start).total_seconds() / 60
        return int(duration)
    except (ValueError, TypeError):
        return 60  # Default if parsing fails 