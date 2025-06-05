"""
Unified Data Library (UDL) Client for SDA Integration
Handles authentication, data access, and collection request/response workflows
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class UDLConfig(BaseModel):
    """Configuration for UDL access"""
    base_url: str = Field(default="https://udl.bluestack.mil")
    api_key: str = Field(...)
    group_id: str = Field(default="sda-tap-lab")
    timeout: int = Field(default=30)
    max_retries: int = Field(default=3)


class CollectionRequest(BaseModel):
    """UDL Collection Request Schema"""
    request_id: str
    sensor_id: str
    target_id: str
    collection_type: str  # OPTICAL, RADAR, RF
    priority: int = Field(ge=1, le=5)
    start_time: datetime
    end_time: datetime
    requirements: Dict[str, Any] = Field(default_factory=dict)


class CollectionResponse(BaseModel):
    """UDL Collection Response Schema"""
    request_id: str
    status: str  # ACCEPTED, REJECTED, SCHEDULED, COMPLETED, FAILED
    sensor_id: str
    scheduled_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    data_url: Optional[str] = None
    rejection_reason: Optional[str] = None


class SensorObservation(BaseModel):
    """Sensor observation data from UDL"""
    observation_id: str
    sensor_id: str
    target_id: str
    timestamp: datetime
    observation_type: str
    position: Dict[str, float]  # lat, lon, alt
    velocity: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TLE(BaseModel):
    """Two-Line Element set from UDL"""
    norad_id: str
    name: str
    line1: str
    line2: str
    epoch: datetime
    classification: str = "U"


class UDLClient:
    """Client for interacting with the Unified Data Library"""
    
    def __init__(self, config: UDLConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._auth_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        
    async def connect(self):
        """Initialize connection to UDL"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        await self._authenticate()
        
    async def disconnect(self):
        """Close UDL connection"""
        if self.session:
            await self.session.close()
            
    async def _authenticate(self):
        """Authenticate with UDL and get access token"""
        try:
            auth_data = {
                "api_key": self.config.api_key,
                "group_id": self.config.group_id
            }
            
            async with self.session.post(
                f"{self.config.base_url}/auth/token",
                json=auth_data
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                self._auth_token = data["access_token"]
                expires_in = data.get("expires_in", 3600)
                self._token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
                
                logger.info("Successfully authenticated with UDL")
                
        except Exception as e:
            logger.error(f"UDL authentication failed: {e}")
            raise
            
    async def _ensure_authenticated(self):
        """Ensure we have a valid auth token"""
        if not self._auth_token or datetime.utcnow() >= self._token_expires:
            await self._authenticate()
            
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to UDL"""
        await self._ensure_authenticated()
        
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self._auth_token}"
        kwargs["headers"] = headers
        
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"UDL request failed after {self.config.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)
                
    async def get_sensor_observations(
        self,
        sensor_id: Optional[str] = None,
        target_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        observation_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[SensorObservation]:
        """Get sensor observations from UDL"""
        params = {
            "limit": limit
        }
        
        if sensor_id:
            params["sensor_id"] = sensor_id
        if target_id:
            params["target_id"] = target_id
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        if observation_type:
            params["observation_type"] = observation_type
            
        data = await self._make_request("GET", "/observations", params=params)
        
        return [SensorObservation(**obs) for obs in data["observations"]]
        
    async def get_tles(
        self,
        norad_ids: Optional[List[str]] = None,
        epoch_start: Optional[datetime] = None,
        epoch_end: Optional[datetime] = None
    ) -> List[TLE]:
        """Get TLEs from UDL"""
        params = {}
        
        if norad_ids:
            params["norad_ids"] = ",".join(norad_ids)
        if epoch_start:
            params["epoch_start"] = epoch_start.isoformat()
        if epoch_end:
            params["epoch_end"] = epoch_end.isoformat()
            
        data = await self._make_request("GET", "/tles", params=params)
        
        return [TLE(**tle) for tle in data["tles"]]
        
    async def submit_collection_request(
        self,
        request: CollectionRequest
    ) -> CollectionResponse:
        """Submit a collection request to UDL"""
        data = await self._make_request(
            "POST",
            "/collection/request",
            json=request.dict()
        )
        
        return CollectionResponse(**data)
        
    async def get_collection_status(
        self,
        request_id: str
    ) -> CollectionResponse:
        """Check status of a collection request"""
        data = await self._make_request(
            "GET",
            f"/collection/request/{request_id}"
        )
        
        return CollectionResponse(**data)
        
    async def get_state_vectors(
        self,
        object_ids: List[str],
        epoch: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get state vectors for objects"""
        params = {
            "object_ids": ",".join(object_ids)
        }
        
        if epoch:
            params["epoch"] = epoch.isoformat()
            
        data = await self._make_request("GET", "/state_vectors", params=params)
        
        return data["state_vectors"]
        
    async def subscribe_to_updates(
        self,
        topic: str,
        callback: callable
    ):
        """Subscribe to real-time UDL updates via WebSocket"""
        ws_url = self.config.base_url.replace("https://", "wss://")
        ws_url = f"{ws_url}/ws"
        
        async with self.session.ws_connect(ws_url) as ws:
            # Authenticate WebSocket
            await ws.send_json({
                "type": "auth",
                "token": self._auth_token
            })
            
            # Subscribe to topic
            await ws.send_json({
                "type": "subscribe",
                "topic": topic
            })
            
            # Listen for messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await callback(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                    

class UDLDataProcessor:
    """Process data from UDL for Welders Arc integration"""
    
    def __init__(self, udl_client: UDLClient):
        self.udl_client = udl_client
        
    async def process_new_observations(self, observations: List[SensorObservation]):
        """Process new sensor observations"""
        for obs in observations:
            # Check if this is an uncorrelated track
            if obs.metadata.get("correlated", False) == False:
                await self._process_uct(obs)
            else:
                await self._process_correlated_observation(obs)
                
    async def _process_uct(self, observation: SensorObservation):
        """Process uncorrelated track"""
        logger.info(f"Processing UCT: {observation.observation_id}")
        # Send to UCT processing pipeline
        
    async def _process_correlated_observation(self, observation: SensorObservation):
        """Process correlated observation"""
        logger.info(f"Processing correlated observation: {observation.observation_id}")
        # Update state vectors, check for events
        
    async def monitor_collection_requests(self, pending_requests: List[str]):
        """Monitor status of pending collection requests"""
        for request_id in pending_requests:
            try:
                response = await self.udl_client.get_collection_status(request_id)
                
                if response.status == "COMPLETED":
                    logger.info(f"Collection request {request_id} completed")
                    # Process collected data
                elif response.status == "FAILED":
                    logger.error(f"Collection request {request_id} failed: {response.rejection_reason}")
                    # Handle failure
                    
            except Exception as e:
                logger.error(f"Error checking collection request {request_id}: {e}") 