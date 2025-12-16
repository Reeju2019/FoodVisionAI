"""
Tests for security features.
"""
import pytest
from fastapi.testclient import TestClient
from foodvision_ai.api.security import validate_image_file


@pytest.mark.unit
def test_security_headers(test_client: TestClient):
    """Test that security headers are present in responses."""
    response = test_client.get("/health")
    
    # Check security headers
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    
    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"
    
    assert "X-XSS-Protection" in response.headers
    assert "Strict-Transport-Security" in response.headers
    assert "Referrer-Policy" in response.headers


@pytest.mark.unit
def test_validate_jpeg_image(sample_image_bytes):
    """Test JPEG image validation."""
    result = validate_image_file(sample_image_bytes, "test.jpg")
    assert result is True
    
    result = validate_image_file(sample_image_bytes, "test.jpeg")
    assert result is True


@pytest.mark.unit
def test_validate_png_image(sample_png_bytes):
    """Test PNG image validation."""
    result = validate_image_file(sample_png_bytes, "test.png")
    assert result is True


@pytest.mark.unit
def test_reject_invalid_extension():
    """Test rejection of invalid file extensions."""
    fake_image = b'\xff\xd8\xff\xe0'  # JPEG magic bytes
    
    result = validate_image_file(fake_image, "test.exe")
    assert result is False
    
    result = validate_image_file(fake_image, "test.txt")
    assert result is False
    
    result = validate_image_file(fake_image, "test.pdf")
    assert result is False


@pytest.mark.unit
def test_reject_invalid_magic_bytes():
    """Test rejection of files with invalid magic bytes."""
    fake_data = b'This is not an image file'
    
    result = validate_image_file(fake_data, "test.jpg")
    assert result is False


@pytest.mark.unit
def test_reject_too_small_file():
    """Test rejection of files that are too small."""
    tiny_file = b'\xff\xd8'  # Only 2 bytes
    
    result = validate_image_file(tiny_file, "test.jpg")
    assert result is False


@pytest.mark.unit
def test_path_traversal_protection(test_client: TestClient):
    """Test protection against path traversal attacks."""
    # Try path traversal in URL
    response = test_client.get("/api/v1/../../../etc/passwd")
    assert response.status_code in [400, 404]


@pytest.mark.unit
def test_xss_protection(test_client: TestClient):
    """Test protection against XSS attacks."""
    # Try XSS in URL
    response = test_client.get("/api/v1/<script>alert('xss')</script>")
    assert response.status_code in [400, 404]

