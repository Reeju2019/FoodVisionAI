"""
Pytest configuration and shared fixtures for FoodVisionAI tests.
"""
import asyncio
import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.testclient import TestClient

from foodvision_ai.api.main import create_app
from foodvision_ai.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def mongodb_client():
    """Create a MongoDB client for testing."""
    client = AsyncIOMotorClient(settings.mongodb_url)
    yield client
    client.close()


@pytest.fixture(scope="function")
async def test_db(mongodb_client):
    """Create a test database and clean it up after each test."""
    db_name = "foodvision_test"
    db = mongodb_client[db_name]
    
    yield db
    
    # Cleanup: drop all collections
    for collection_name in await db.list_collection_names():
        await db[collection_name].drop()


@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_image_bytes():
    """Provide sample image bytes for testing."""
    # Minimal valid JPEG (1x1 pixel)
    return (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c'
        b'\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c'
        b'\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00'
        b'\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01'
        b'\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05'
        b'\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04'
        b'\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A'
        b'\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82'
        b'\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz'
        b'\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a'
        b'\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9'
        b'\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8'
        b'\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5'
        b'\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xfe\x8a'
        b'(\xa2\x8a\xff\xd9'
    )


@pytest.fixture
def sample_png_bytes():
    """Provide sample PNG image bytes for testing."""
    # Minimal valid PNG (1x1 pixel)
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
        b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pytest.fixture
def sample_analysis_data():
    """Provide sample analysis data for testing."""
    return {
        "image_id": "test-image-123",
        "status": "completed",
        "stage1_ingredients": {
            "ingredients": [
                {"name": "rice", "confidence": 0.95},
                {"name": "chicken", "confidence": 0.89}
            ],
            "detection_method": "test_model"
        },
        "stage2_dish_analysis": {
            "predicted_dish": "Chicken Rice",
            "description": "A simple chicken and rice dish",
            "cuisine_type": "Asian",
            "confidence": 0.92
        },
        "stage3_nutrition": {
            "calories": 450,
            "protein_g": 28,
            "fat_g": 15,
            "carbs_g": 52
        }
    }


@pytest.fixture
def mock_gemini_response():
    """Provide mock Gemini API response."""
    return {
        "predicted_dish": "Chicken Curry with Rice",
        "description": "A flavorful curry dish",
        "cuisine_type": "Indian",
        "confidence": 0.92
    }


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_db: Tests requiring database")
    config.addinivalue_line("markers", "requires_models: Tests requiring ML models")
    config.addinivalue_line("markers", "requires_api: Tests requiring external API")

