# Testing Guide for FoodVisionAI

This guide covers testing strategies, running tests, and contributing test coverage.

---

## 📋 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_health.py           # Health check endpoint tests
├── test_security.py         # Security feature tests
├── test_api_endpoints.py    # API endpoint tests
├── test_database.py         # Database operation tests
├── test_models.py           # ML model tests
└── test_pipeline.py         # Pipeline integration tests
```

---

## 🚀 Running Tests

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=foodvision_ai --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# End-to-end tests
pytest -m e2e

# Tests that don't require database
pytest -m "not requires_db"

# Tests that don't require ML models
pytest -m "not requires_models"
```

### Run Specific Test Files

```bash
# Run health check tests
pytest tests/test_health.py

# Run security tests
pytest tests/test_security.py

# Run specific test function
pytest tests/test_health.py::test_basic_health_check
```

---

## 🏷️ Test Markers

Tests are categorized using pytest markers:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Unit tests for individual components |
| `@pytest.mark.integration` | Integration tests for multiple components |
| `@pytest.mark.e2e` | End-to-end tests |
| `@pytest.mark.slow` | Tests that take a long time |
| `@pytest.mark.requires_db` | Tests requiring database connection |
| `@pytest.mark.requires_models` | Tests requiring ML models |
| `@pytest.mark.requires_api` | Tests requiring external API access |

### Example Usage

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    """A simple unit test."""
    assert 1 + 1 == 2

@pytest.mark.integration
@pytest.mark.requires_db
async def test_database_operation(test_db):
    """An integration test requiring database."""
    result = await test_db.collection.insert_one({"test": "data"})
    assert result.inserted_id is not None
```

---

## 🔧 Test Fixtures

Common fixtures available in `conftest.py`:

### Database Fixtures

```python
async def test_with_database(test_db):
    """Use test database."""
    # test_db is automatically cleaned up after test
    await test_db.collection.insert_one({"key": "value"})
```

### API Client Fixture

```python
def test_api_endpoint(test_client):
    """Use FastAPI test client."""
    response = test_client.get("/health")
    assert response.status_code == 200
```

### Sample Data Fixtures

```python
def test_image_upload(sample_image_bytes):
    """Use sample JPEG image."""
    # sample_image_bytes contains valid JPEG data
    assert len(sample_image_bytes) > 0

def test_png_upload(sample_png_bytes):
    """Use sample PNG image."""
    # sample_png_bytes contains valid PNG data
    assert len(sample_png_bytes) > 0
```

---

## 📊 Coverage Reports

### Generate HTML Coverage Report

```bash
# Run tests with coverage
pytest --cov=foodvision_ai --cov-report=html

# Open report in browser
# Windows:
start htmlcov/index.html

# Linux/Mac:
open htmlcov/index.html
```

### Generate Terminal Coverage Report

```bash
pytest --cov=foodvision_ai --cov-report=term-missing
```

### Coverage Goals

- **Overall**: > 80%
- **Critical paths**: > 90%
- **API endpoints**: > 85%
- **Database operations**: > 85%

---

## 🔄 Continuous Integration

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

### CI Pipeline

1. **Lint**: Code style and formatting checks
2. **Security**: Security vulnerability scanning
3. **Unit Tests**: Fast, isolated tests
4. **Integration Tests**: Tests with database
5. **Docker Build**: Verify Docker image builds

See `.github/workflows/ci.yml` for details.

---

## ✍️ Writing Tests

### Test Naming Convention

```python
# Good test names
def test_health_check_returns_200():
    """Test that health check returns 200 status."""
    pass

def test_upload_rejects_invalid_file():
    """Test that upload rejects non-image files."""
    pass

# Bad test names
def test_1():
    pass

def test_stuff():
    pass
```

### Test Structure (AAA Pattern)

```python
def test_example():
    # Arrange - Set up test data
    data = {"key": "value"}
    
    # Act - Perform the action
    result = process_data(data)
    
    # Assert - Verify the result
    assert result["status"] == "success"
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_operation()
    assert result is not None
```

---

## 🐛 Debugging Tests

### Run with Debug Output

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb
```

### Run Failed Tests Only

```bash
# Run only tests that failed last time
pytest --lf

# Run failed tests first, then others
pytest --ff
```

---

## 📝 Best Practices

1. **Keep tests independent** - Each test should run in isolation
2. **Use fixtures** - Reuse common setup code
3. **Test edge cases** - Don't just test happy paths
4. **Mock external services** - Don't rely on external APIs in tests
5. **Keep tests fast** - Mark slow tests with `@pytest.mark.slow`
6. **Write descriptive assertions** - Use clear error messages
7. **Clean up resources** - Use fixtures for setup/teardown

---

## 🔍 Common Issues

### Database Connection Errors

```bash
# Make sure MongoDB is running
docker run -d -p 27017:27017 mongo:7.0

# Or use docker-compose
docker-compose up -d mongodb
```

### Import Errors

```bash
# Install package in development mode
pip install -e .
```

### Async Test Errors

```bash
# Make sure pytest-asyncio is installed
pip install pytest-asyncio
```

---

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Motor Testing](https://motor.readthedocs.io/en/stable/tutorial-asyncio.html)

---

**Last Updated**: 2025-12-15  
**Version**: 1.0.0

