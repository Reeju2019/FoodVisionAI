"""
Tests for health check endpoints.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_basic_health_check(test_client: TestClient):
    """Test basic health check endpoint."""
    response = test_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"
    assert data["service"] == "FoodVisionAI"


@pytest.mark.unit
def test_liveness_check(test_client: TestClient):
    """Test liveness check endpoint."""
    response = test_client.get("/health/live")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "alive"
    assert "timestamp" in data
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


@pytest.mark.integration
@pytest.mark.requires_db
def test_readiness_check(test_client: TestClient):
    """Test readiness check endpoint."""
    response = test_client.get("/health/ready")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should be ready if database is available
    assert "status" in data
    assert data["status"] in ["ready", "not_ready"]
    assert "timestamp" in data


@pytest.mark.integration
@pytest.mark.requires_db
def test_detailed_health_check(test_client: TestClient):
    """Test detailed health check endpoint."""
    response = test_client.get("/health/detailed")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "uptime_seconds" in data
    assert "components" in data
    
    # Check components
    components = data["components"]
    assert "database" in components or "system" in components


@pytest.mark.integration
@pytest.mark.requires_db
def test_metrics_endpoint(test_client: TestClient):
    """Test metrics endpoint."""
    response = test_client.get("/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "timestamp" in data
    assert "uptime_seconds" in data
    assert "application" in data
    
    # Check application info
    app_info = data["application"]
    assert app_info["name"] == "FoodVisionAI"
    assert "version" in app_info

