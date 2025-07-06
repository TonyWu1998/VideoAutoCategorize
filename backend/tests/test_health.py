"""
Test cases for health check endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the main health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "uptime_seconds" in data


def test_simple_health_check():
    """Test the simple health check endpoint."""
    response = client.get("/api/health/simple")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["message"] == "Media Semantic Search API"
    assert data["version"] == "1.0.0"


def test_api_info_endpoint():
    """Test the API info endpoint."""
    response = client.get("/api")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "Media Semantic Search API"
    assert "endpoints" in data
    assert "documentation" in data
