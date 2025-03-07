"""Unified Data Library (UDL) Service for AstroShield.

This service provides integration with the Unified Data Library (UDL),
a repository aggregating Space Domain Awareness (SDA) data from multiple sources.
"""

import logging
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from infrastructure.circuit_breaker import circuit_breaker
from infrastructure.monitoring import MonitoringService
from infrastructure.cache import CacheManager

logger = logging.getLogger(__name__)
monitoring = MonitoringService()
cache = CacheManager()

class UDLService:
    """Service for interacting with the Unified Data Library (UDL)."""
    
    def __init__(self, base_url: str, api_key: str):
        """Initialize the UDL service.
        
        Args:
            base_url: Base URL for the UDL API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.session = None
    
    async def initialize(self):
        """Initialize the HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @circuit_breaker
    async def get_spacecraft_data(self, spacecraft_id: str) -> Dict[str, Any]:
        """Get spacecraft data from UDL.
        
        Args:
            spacecraft_id: ID of the spacecraft
            
        Returns:
            Dictionary containing spacecraft data
        """
        with monitoring.create_span("udl_get_spacecraft_data") as span:
            span.set_attribute("spacecraft_id", spacecraft_id)
            
            # Check cache first
            cache_key = f"udl_spacecraft_{spacecraft_id}"
            cached_data = await cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved spacecraft data from cache for {spacecraft_id}")
                return cached_data
            
            await self.initialize()
            try:
                url = f"{self.base_url}/objects/{spacecraft_id}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the result
                        await cache.set(cache_key, data, ttl=timedelta(hours=1))
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Error retrieving spacecraft data: {error_text}")
                        return {"error": f"HTTP {response.status}: {error_text}"}
            except Exception as e:
                logger.error(f"Error connecting to UDL: {str(e)}")
                return {"error": str(e)}
    
    @circuit_breaker
    async def search_objects(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for space objects in UDL.
        
        Args:
            query: Search parameters
            
        Returns:
            List of matching space objects
        """
        with monitoring.create_span("udl_search_objects") as span:
            span.set_attribute("query", str(query))
            
            # Generate cache key based on query parameters
            cache_key = f"udl_search_{hash(frozenset(query.items()))}"
            cached_data = await cache.get(cache_key)
            if cached_data:
                logger.info("Retrieved search results from cache")
                return cached_data
            
            await self.initialize()
            try:
                url = f"{self.base_url}/objects/search"
                async with self.session.post(url, json=query) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the result
                        await cache.set(cache_key, data, ttl=timedelta(minutes=30))
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Error searching objects: {error_text}")
                        return []
            except Exception as e:
                logger.error(f"Error connecting to UDL: {str(e)}")
                return []
    
    @circuit_breaker
    async def get_conjunction_data(self, spacecraft_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get conjunction data for a spacecraft.
        
        Args:
            spacecraft_id: ID of the spacecraft
            days: Number of days to look ahead
            
        Returns:
            List of conjunction events
        """
        with monitoring.create_span("udl_get_conjunction_data") as span:
            span.set_attribute("spacecraft_id", spacecraft_id)
            span.set_attribute("days", days)
            
            # Check cache first
            cache_key = f"udl_conjunction_{spacecraft_id}_{days}"
            cached_data = await cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved conjunction data from cache for {spacecraft_id}")
                return cached_data
            
            await self.initialize()
            try:
                url = f"{self.base_url}/conjunctions"
                params = {
                    "object_id": spacecraft_id,
                    "days": days
                }
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the result
                        await cache.set(cache_key, data, ttl=timedelta(hours=1))
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Error retrieving conjunction data: {error_text}")
                        return []
            except Exception as e:
                logger.error(f"Error connecting to UDL: {str(e)}")
                return []
    
    @circuit_breaker
    async def get_tle_data(self, spacecraft_id: str) -> Dict[str, Any]:
        """Get TLE data for a spacecraft.
        
        Args:
            spacecraft_id: ID of the spacecraft
            
        Returns:
            Dictionary containing TLE data
        """
        with monitoring.create_span("udl_get_tle_data") as span:
            span.set_attribute("spacecraft_id", spacecraft_id)
            
            # Check cache first
            cache_key = f"udl_tle_{spacecraft_id}"
            cached_data = await cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved TLE data from cache for {spacecraft_id}")
                return cached_data
            
            await self.initialize()
            try:
                url = f"{self.base_url}/tle/{spacecraft_id}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the result
                        await cache.set(cache_key, data, ttl=timedelta(hours=6))
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Error retrieving TLE data: {error_text}")
                        return {"error": f"HTTP {response.status}: {error_text}"}
            except Exception as e:
                logger.error(f"Error connecting to UDL: {str(e)}")
                return {"error": str(e)}
    
    @circuit_breaker
    async def get_ephemeris(self, spacecraft_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get ephemeris data for a spacecraft.
        
        Args:
            spacecraft_id: ID of the spacecraft
            start_time: Start time for ephemeris
            end_time: End time for ephemeris
            
        Returns:
            List of ephemeris points
        """
        with monitoring.create_span("udl_get_ephemeris") as span:
            span.set_attribute("spacecraft_id", spacecraft_id)
            
            # Check cache first
            cache_key = f"udl_ephemeris_{spacecraft_id}_{start_time.isoformat()}_{end_time.isoformat()}"
            cached_data = await cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved ephemeris data from cache for {spacecraft_id}")
                return cached_data
            
            await self.initialize()
            try:
                url = f"{self.base_url}/ephemeris/{spacecraft_id}"
                params = {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the result
                        await cache.set(cache_key, data, ttl=timedelta(hours=1))
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Error retrieving ephemeris data: {error_text}")
                        return []
            except Exception as e:
                logger.error(f"Error connecting to UDL: {str(e)}")
                return [] 