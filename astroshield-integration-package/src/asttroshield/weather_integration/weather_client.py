"""
Weather data client for fetching weather conditions.
"""
import aiohttp
import asyncio
import datetime
from typing import Dict, Any, Optional, List

class WeatherDataClient:
    """Client for fetching weather data for observation planning."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.weatherapi.com/v1"):
        """
        Initialize the weather data client.
        
        Args:
            api_key: API key for the weather service
            base_url: Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_current_weather(self, location: str) -> Dict[str, Any]:
        """
        Get current weather conditions for a location.
        
        Args:
            location: Location string (e.g., "New York" or "40.7128,-74.0060")
            
        Returns:
            Weather data dictionary
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        params = {
            "key": self.api_key,
            "q": location,
        }
        
        url = f"{self.base_url}/current.json"
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._transform_current_weather(data)
            else:
                raise Exception(f"Error fetching weather data: {response.status}")
    
    async def get_forecast(self, location: str, days: int = 1) -> Dict[str, Any]:
        """
        Get weather forecast for a location.
        
        Args:
            location: Location string (e.g., "New York" or "40.7128,-74.0060")
            days: Number of days to forecast (1-10)
            
        Returns:
            Weather forecast dictionary
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        params = {
            "key": self.api_key,
            "q": location,
            "days": min(10, max(1, days)),
            "aqi": "yes",
        }
        
        url = f"{self.base_url}/forecast.json"
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._transform_forecast(data)
            else:
                raise Exception(f"Error fetching forecast data: {response.status}")
    
    async def get_astronomy(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get astronomical data for a location (sunrise, sunset, etc.)
        
        Args:
            location: Location string (e.g., "New York" or "40.7128,-74.0060")
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Astronomy data dictionary
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        if not date:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        params = {
            "key": self.api_key,
            "q": location,
            "dt": date,
        }
        
        url = f"{self.base_url}/astronomy.json"
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._transform_astronomy(data)
            else:
                raise Exception(f"Error fetching astronomy data: {response.status}")
    
    def _transform_current_weather(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw weather data to our format."""
        location = data.get("location", {})
        current = data.get("current", {})
        
        return {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "location": {
                "name": location.get("name"),
                "region": location.get("region"),
                "country": location.get("country"),
                "latitude": location.get("lat"),
                "longitude": location.get("lon"),
            },
            "conditions": {
                "temperature": {
                    "celsius": current.get("temp_c"),
                    "fahrenheit": current.get("temp_f"),
                },
                "feelsLike": {
                    "celsius": current.get("feelslike_c"),
                    "fahrenheit": current.get("feelslike_f"),
                },
                "windSpeed": {
                    "kph": current.get("wind_kph"),
                    "mph": current.get("wind_mph"),
                },
                "windDirection": {
                    "degrees": current.get("wind_degree"),
                    "direction": current.get("wind_dir"),
                },
                "precipitation": {
                    "mm": current.get("precip_mm"),
                    "in": current.get("precip_in"),
                },
                "humidity": current.get("humidity"),
                "clouds": {
                    "coverage": current.get("cloud"),
                },
                "visibility": {
                    "km": current.get("vis_km"),
                    "miles": current.get("vis_miles"),
                },
                "uv": current.get("uv"),
                "condition": {
                    "text": current.get("condition", {}).get("text"),
                    "code": current.get("condition", {}).get("code"),
                }
            }
        }
    
    def _transform_forecast(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw forecast data to our format."""
        location = data.get("location", {})
        forecast = data.get("forecast", {})
        forecast_days = forecast.get("forecastday", [])
        
        days = []
        for day in forecast_days:
            date = day.get("date")
            day_data = day.get("day", {})
            astro = day.get("astro", {})
            
            # Extract hour-by-hour forecast
            hours = []
            for hour in day.get("hour", []):
                hour_time = hour.get("time")
                hour_data = {
                    "time": hour_time,
                    "temperature": {
                        "celsius": hour.get("temp_c"),
                        "fahrenheit": hour.get("temp_f"),
                    },
                    "clouds": {
                        "coverage": hour.get("cloud"),
                    },
                    "visibility": {
                        "km": hour.get("vis_km"),
                        "miles": hour.get("vis_miles"),
                    },
                    "precipitation": {
                        "mm": hour.get("precip_mm"),
                        "in": hour.get("precip_in"),
                        "chance": hour.get("chance_of_rain"),
                    },
                    "condition": {
                        "text": hour.get("condition", {}).get("text"),
                        "code": hour.get("condition", {}).get("code"),
                    }
                }
                hours.append(hour_data)
            
            # Create day data
            day_forecast = {
                "date": date,
                "temperature": {
                    "min_celsius": day_data.get("mintemp_c"),
                    "max_celsius": day_data.get("maxtemp_c"),
                    "min_fahrenheit": day_data.get("mintemp_f"),
                    "max_fahrenheit": day_data.get("maxtemp_f"),
                },
                "astronomy": {
                    "sunrise": astro.get("sunrise"),
                    "sunset": astro.get("sunset"),
                    "moonrise": astro.get("moonrise"),
                    "moonset": astro.get("moonset"),
                    "moon_phase": astro.get("moon_phase"),
                    "moon_illumination": astro.get("moon_illumination"),
                },
                "condition": {
                    "text": day_data.get("condition", {}).get("text"),
                    "code": day_data.get("condition", {}).get("code"),
                },
                "hourly": hours
            }
            days.append(day_forecast)
            
        return {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "location": {
                "name": location.get("name"),
                "region": location.get("region"),
                "country": location.get("country"),
                "latitude": location.get("lat"),
                "longitude": location.get("lon"),
            },
            "forecast_period": {
                "start": forecast_days[0].get("date") if forecast_days else None,
                "end": forecast_days[-1].get("date") if forecast_days else None,
            },
            "days": days
        }
    
    def _transform_astronomy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw astronomy data to our format."""
        location = data.get("location", {})
        astronomy = data.get("astronomy", {})
        astro = astronomy.get("astro", {})
        
        return {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "location": {
                "name": location.get("name"),
                "region": location.get("region"),
                "country": location.get("country"),
                "latitude": location.get("lat"),
                "longitude": location.get("lon"),
            },
            "date": data.get("astronomy", {}).get("date"),
            "astronomy": {
                "sunrise": astro.get("sunrise"),
                "sunset": astro.get("sunset"),
                "moonrise": astro.get("moonrise"),
                "moonset": astro.get("moonset"),
                "moon_phase": astro.get("moon_phase"),
                "moon_illumination": astro.get("moon_illumination"),
            }
        } 