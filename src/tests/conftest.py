"""Pytest configuration and shared fixtures."""

import os
import sys
import pytest
from pathlib import Path
from asttroshield.api_client.udl_client import UDLClient

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Configure test environment
os.environ['TESTING'] = 'true'
os.environ['UDL_API_KEY'] = 'test-key'
os.environ['UDL_BASE_URL'] = 'https://unifieddatalibrary.com/udl'

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def udl_client():
    """Create a UDL client for testing."""
    return UDLClient(
        base_url=os.environ['UDL_BASE_URL'],
        api_key=os.environ['UDL_API_KEY']
    )

@pytest.fixture
def mock_response():
    """Create a mock response object."""
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            
        def json(self):
            return self.json_data
            
    return MockResponse 