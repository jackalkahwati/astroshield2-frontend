import pytest
from flask import url_for
from app import create_app
from models import db

@pytest.fixture
def app():
    """Create and configure a test Flask application instance."""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create a test CLI runner."""
    return app.test_cli_runner()

@pytest.fixture
def init_database(app):
    """Initialize test database."""
    with app.app_context():
        db.create_all()
        yield db
        db.session.remove()
        db.drop_all()

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/system/health')
    assert response.status_code == 200
    assert response.json['status'] == 'operational'

def test_spacecraft_status(client, init_database):
    """Test getting spacecraft status."""
    spacecraft_id = "test_spacecraft_1"
    response = client.get(f'/spacecraft/{spacecraft_id}/status')
    assert response.status_code == 200
    assert 'status' in response.json

def test_analyze_conjunction(client, init_database):
    """Test conjunction analysis endpoint."""
    spacecraft_id = "test_spacecraft_1"
    other_spacecraft_id = "test_spacecraft_2"
    response = client.get(f'/spacecraft/{spacecraft_id}/conjunction/{other_spacecraft_id}')
    assert response.status_code == 200
    assert 'risk_assessment' in response.json

def test_get_threats(client, init_database):
    """Test getting spacecraft threats."""
    spacecraft_id = "test_spacecraft_1"
    response = client.get(f'/spacecraft/{spacecraft_id}/threats')
    assert response.status_code == 200
    assert isinstance(response.json['threats'], list)

def test_deploy_countermeasures(client, init_database):
    """Test deploying countermeasures."""
    spacecraft_id = "test_spacecraft_1"
    payload = {
        'countermeasure_type': 'evasive_maneuver',
        'parameters': {
            'delta_v': 1.5,
            'direction': [1, 0, 0]
        }
    }
    response = client.post(f'/spacecraft/{spacecraft_id}/countermeasures', json=payload)
    assert response.status_code == 200
    assert 'deployment_status' in response.json

def test_invalid_spacecraft_id(client):
    """Test response with invalid spacecraft ID."""
    response = client.get('/spacecraft/invalid_id/status')
    assert response.status_code == 404

def test_rate_limiting(client):
    """Test rate limiting functionality."""
    # Make multiple rapid requests
    responses = [
        client.get('/system/health')
        for _ in range(100)  # Adjust based on your rate limit
    ]
    # At least one should be rate limited
    assert any(r.status_code == 429 for r in responses) 