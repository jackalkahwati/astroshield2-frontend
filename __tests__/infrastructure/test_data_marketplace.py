"""
Test suite for Data Marketplace Integration module
"""

import pytest
import aiohttp
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from infrastructure.data_marketplace import (
    DataMarketplaceIntegration,
    MarketplaceConfig,
    DataProduct
)

@pytest.fixture
def config():
    return MarketplaceConfig(
        gdm_endpoint="https://gdm.sda.mil/api/v1",
        udl_endpoint="https://udl.sda.mil/api/v1",
        api_key="test-api-key",
        organization_id="test-org",
        data_classification="CUI",
        export_control="ITAR"
    )

@pytest.fixture
def mock_kill_chain_event():
    return {
        'event_type': 'MANEUVER',
        'confidence': 0.95,
        'timestamp': datetime.now(),
        'source_object': 'SAT123',
        'evidence': {
            'delta_v': 0.15,
            'maneuver_type': 'INCLINATION_CHANGE'
        }
    }

@pytest.fixture
def mock_ccdm_indicator():
    return {
        'indicator_name': 'orbit_anomaly',
        'confidence_level': 0.88,
        'timestamp': datetime.now(),
        'evidence': {
            'orbital_change': True,
            'deviation_magnitude': 0.5
        }
    }

class MockResponse:
    def __init__(self, status, json_data):
        self.status = status
        self._json = json_data

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_marketplace_initialization(config):
    """Test marketplace client initialization"""
    async with DataMarketplaceIntegration(config) as client:
        assert client.config == config
        assert client._session is not None
        
        # Verify session headers
        assert client._session._default_headers['Authorization'] == f"Bearer {config.api_key}"
        assert client._session._default_headers['X-Organization-ID'] == config.organization_id

@pytest.mark.asyncio
async def test_publish_kill_chain_events(config, mock_kill_chain_event):
    """Test publishing kill chain events to marketplace"""
    async with DataMarketplaceIntegration(config) as client:
        with patch.object(client._session, 'post', 
                         return_value=MockResponse(201, {'product_id': 'test-product-1'})), \
             patch.object(client._session, 'put',
                         return_value=MockResponse(200, {})):
            
            # Publish event
            product_ids = await client.publish_kill_chain_events([mock_kill_chain_event])
            
            # Verify results
            assert len(product_ids) == 1
            assert product_ids[0] == 'test-product-1'

@pytest.mark.asyncio
async def test_publish_ccdm_indicators(config, mock_ccdm_indicator):
    """Test publishing CCDM indicators to marketplace"""
    async with DataMarketplaceIntegration(config) as client:
        with patch.object(client._session, 'post',
                         return_value=MockResponse(201, {'product_id': 'test-product-2'})), \
             patch.object(client._session, 'put',
                         return_value=MockResponse(200, {})):
            
            # Publish indicator
            product_ids = await client.publish_ccdm_indicators([mock_ccdm_indicator])
            
            # Verify results
            assert len(product_ids) == 1
            assert product_ids[0] == 'test-product-2'

@pytest.mark.asyncio
async def test_query_marketplace(config):
    """Test querying marketplace for products"""
    async with DataMarketplaceIntegration(config) as client:
        mock_product = {
            'product_id': 'test-product-3',
            'name': 'Test Product',
            'description': 'Test Description',
            'data_type': 'KILL_CHAIN_EVENT',
            'classification': 'CUI',
            'export_control': 'ITAR',
            'timestamp': datetime.now().isoformat(),
            'content': {'test': 'data'},
            'metadata': {'source': 'test'}
        }
        
        with patch.object(client._session, 'get',
                         return_value=MockResponse(200, [mock_product])):
            
            # Query marketplace
            query_params = {
                'data_type': 'KILL_CHAIN_EVENT',
                'start_time': datetime.now().isoformat()
            }
            products = await client.query_marketplace(query_params)
            
            # Verify results
            assert len(products) == 1
            assert products[0].product_id == 'test-product-3'
            assert products[0].data_type == 'KILL_CHAIN_EVENT'

@pytest.mark.asyncio
async def test_error_handling(config, mock_kill_chain_event):
    """Test error handling in marketplace operations"""
    async with DataMarketplaceIntegration(config) as client:
        with patch.object(client._session, 'post',
                         return_value=MockResponse(500, {})):
            
            # Attempt to publish
            product_ids = await client.publish_kill_chain_events([mock_kill_chain_event])
            
            # Verify empty result on error
            assert len(product_ids) == 0

@pytest.mark.asyncio
async def test_data_validation(config):
    """Test data product validation"""
    # Test valid product
    valid_product = DataProduct(
        product_id="test-id",
        name="Test Product",
        description="Test Description",
        data_type="KILL_CHAIN_EVENT",
        classification="CUI",
        export_control="ITAR",
        timestamp=datetime.now(),
        content={'test': 'data'},
        metadata={'source': 'test'}
    )
    assert valid_product is not None
    
    # Test invalid product (should raise validation error)
    with pytest.raises(ValueError):
        DataProduct(
            product_id="test-id",
            name="Test Product",
            # Missing required fields
            timestamp=datetime.now()
        )
