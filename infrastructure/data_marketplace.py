"""
Data Marketplace Integration Module for AstroShield
Handles integration with SDA's Global Data Marketplace (GDM) and Unified Data Library (UDL)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import json
import logging
from pydantic import BaseModel
from infrastructure.monitoring import MonitoringService

logger = logging.getLogger(__name__)
monitoring = MonitoringService()

class MarketplaceConfig(BaseModel):
    """Configuration for data marketplace integration"""
    gdm_endpoint: str
    udl_endpoint: str
    api_key: str
    organization_id: str
    data_classification: str
    export_control: str

class DataProduct(BaseModel):
    """Model for data products in the marketplace"""
    product_id: str
    name: str
    description: str
    data_type: str
    classification: str
    export_control: str
    timestamp: datetime
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]]

class DataMarketplaceIntegration:
    """Handles integration with SDA data marketplaces"""
    
    def __init__(self, config: MarketplaceConfig):
        self.config = config
        self._session = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "X-Organization-ID": self.config.organization_id
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def publish_kill_chain_events(self, 
                                      events: List[Dict[str, Any]]) -> List[str]:
        """Publish kill chain events to the marketplace"""
        try:
            with monitoring.create_span("publish_kill_chain_events") as span:
                product_ids = []
                
                for event in events:
                    product = DataProduct(
                        product_id=f"kce_{event['source_object']}_{datetime.now().timestamp()}",
                        name=f"Kill Chain Event - {event['event_type']}",
                        description=f"Detection of {event['event_type']} event for object {event['source_object']}",
                        data_type="KILL_CHAIN_EVENT",
                        classification=self.config.data_classification,
                        export_control=self.config.export_control,
                        timestamp=event['timestamp'],
                        content=event,
                        metadata={
                            "confidence": event['confidence'],
                            "event_type": event['event_type'],
                            "source": "AstroShield"
                        }
                    )
                    
                    # Publish to GDM
                    gdm_response = await self._publish_to_gdm(product)
                    if gdm_response:
                        product_ids.append(gdm_response)
                    
                    # Sync with UDL
                    await self._sync_with_udl(product)
                
                span.set_attribute("published_events", len(product_ids))
                return product_ids
                
        except Exception as e:
            logger.error(f"Error publishing kill chain events: {str(e)}")
            raise
    
    async def publish_ccdm_indicators(self, 
                                    indicators: List[Dict[str, Any]]) -> List[str]:
        """Publish CCDM indicators to the marketplace"""
        try:
            with monitoring.create_span("publish_ccdm_indicators") as span:
                product_ids = []
                
                for indicator in indicators:
                    product = DataProduct(
                        product_id=f"ccdm_{indicator['indicator_name']}_{datetime.now().timestamp()}",
                        name=f"CCDM Indicator - {indicator['indicator_name']}",
                        description=f"Detection of {indicator['indicator_name']} indicator",
                        data_type="CCDM_INDICATOR",
                        classification=self.config.data_classification,
                        export_control=self.config.export_control,
                        timestamp=indicator['timestamp'],
                        content=indicator,
                        metadata={
                            "confidence": indicator['confidence_level'],
                            "indicator_type": indicator['indicator_name'],
                            "source": "AstroShield"
                        }
                    )
                    
                    # Publish to GDM
                    gdm_response = await self._publish_to_gdm(product)
                    if gdm_response:
                        product_ids.append(gdm_response)
                    
                    # Sync with UDL
                    await self._sync_with_udl(product)
                
                span.set_attribute("published_indicators", len(product_ids))
                return product_ids
                
        except Exception as e:
            logger.error(f"Error publishing CCDM indicators: {str(e)}")
            raise
    
    async def _publish_to_gdm(self, product: DataProduct) -> Optional[str]:
        """Publish data product to Global Data Marketplace"""
        try:
            async with self._session.post(
                f"{self.config.gdm_endpoint}/products",
                json=product.model_dump()
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    return result.get('product_id')
                else:
                    logger.error(f"GDM publish failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error in GDM publish: {str(e)}")
            return None
    
    async def _sync_with_udl(self, product: DataProduct) -> bool:
        """Synchronize data product with Unified Data Library"""
        try:
            async with self._session.put(
                f"{self.config.udl_endpoint}/sync",
                json=product.model_dump()
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error in UDL sync: {str(e)}")
            return False
    
    async def query_marketplace(self, 
                              query_params: Dict[str, Any]) -> List[DataProduct]:
        """Query the data marketplace for relevant products"""
        try:
            async with self._session.get(
                f"{self.config.gdm_endpoint}/products",
                params=query_params
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    return [DataProduct(**product) for product in results]
                else:
                    logger.error(f"Marketplace query failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error in marketplace query: {str(e)}")
            return []
