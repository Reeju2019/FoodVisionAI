# FoodVisionAI API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API key authentication.

---

## Endpoints

### 1. Health Check

Check if the API is running and healthy.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-15T10:30:00Z",
  "version": "1.0.0"
}
```

---

### 2. Upload Image

Upload a food image for analysis.

**Endpoint**: `POST /api/v1/upload`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): Image file (JPEG, PNG)
- `max_size`: 10MB

**Request Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@food_image.jpg"
```

**Response**:
```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "message": "Image uploaded successfully. Analysis pipeline will begin shortly.",
  "analytics_url": "/analytics/550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid file format or size
- `500`: Server error

---

### 3. Get Analysis Status

Check the processing status of an uploaded image.

**Endpoint**: `GET /api/v1/status/{image_id}`

**Parameters**:
- `image_id` (path): UUID of the uploaded image

**Request Example**:
```bash
curl -X GET "http://localhost:8000/api/v1/status/550e8400-e29b-41d4-a716-446655440000"
```

**Response**:
```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": {
    "stage1_ingredients": "completed",
    "stage2_dish_analysis": "in_progress",
    "stage3_nutrition": "pending"
  },
  "created_at": "2025-12-15T10:30:00Z",
  "updated_at": "2025-12-15T10:30:15Z"
}
```

**Status Values**:
- `uploaded`: Image uploaded, waiting to process
- `processing`: Analysis in progress
- `completed`: Analysis finished successfully
- `failed`: Analysis failed

---

### 4. Get Analysis Results

Retrieve complete analysis results for an image.

**Endpoint**: `GET /api/v1/analytics/{image_id}`

**Parameters**:
- `image_id` (path): UUID of the uploaded image

**Request Example**:
```bash
curl -X GET "http://localhost:8000/api/v1/analytics/550e8400-e29b-41d4-a716-446655440000"
```

**Response**:
```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "image_url": "http://example.com/image.jpg",
  "stage1_ingredients": {
    "ingredients": [
      {"name": "rice", "confidence": 0.95},
      {"name": "chicken", "confidence": 0.89},
      {"name": "curry", "confidence": 0.87}
    ],
    "detection_method": "recipe1m_cnn",
    "timestamp": "2025-12-15T10:30:05Z"
  },
  "stage2_dish_analysis": {
    "predicted_dish": "Chicken Curry with Rice",
    "description": "A flavorful curry dish with tender chicken pieces served over steamed rice",
    "cuisine_type": "Indian",
    "confidence": 0.92,
    "analysis_method": "gemini_pro",
    "timestamp": "2025-12-15T10:30:10Z"
  },
  "stage3_nutrition": {
    "calories": 450,
    "protein_g": 28,
    "fat_g": 15,
    "carbs_g": 52,
    "fiber_g": 3,
    "lookup_method": "database",
    "portion_size": "1 serving (350g)",
    "timestamp": "2025-12-15T10:30:12Z"
  },
  "model_remarks": [
    {
      "component": "stage1_vision",
      "status": "success",
      "message": "Detected 3 ingredients with high confidence",
      "timestamp": "2025-12-15T10:30:05Z"
    },
    {
      "component": "stage2_gemini",
      "status": "success",
      "message": "Dish identified as Chicken Curry with Rice",
      "timestamp": "2025-12-15T10:30:10Z"
    },
    {
      "component": "stage3_nutrition",
      "status": "success",
      "message": "Nutrition lookup found 450 calories using database",
      "timestamp": "2025-12-15T10:30:12Z"
    }
  ],
  "created_at": "2025-12-15T10:30:00Z",
  "completed_at": "2025-12-15T10:30:12Z"
}
```

---

## Interactive API Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-12-15T10:30:00Z"
}
```

### Common Error Codes

- `400`: Bad Request - Invalid input
- `404`: Not Found - Resource doesn't exist
- `422`: Validation Error - Invalid data format
- `500`: Internal Server Error - Server-side error
- `503`: Service Unavailable - Service temporarily down

---

## Rate Limiting

Currently no rate limiting is implemented. For production:

- Recommended: 100 requests per minute per IP
- Burst: 20 requests per second

---

## Best Practices

1. **Poll Status**: Use `/status/{image_id}` to check progress
2. **Error Handling**: Always check response status codes
3. **Timeouts**: Set reasonable timeouts (30-60 seconds)
4. **File Size**: Keep images under 10MB
5. **Format**: Use JPEG or PNG formats

---

## Example Workflow

```python
import requests
import time

# 1. Upload image
with open('food.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/upload',
        files={'file': f}
    )
    data = response.json()
    image_id = data['image_id']

# 2. Poll for status
while True:
    response = requests.get(
        f'http://localhost:8000/api/v1/status/{image_id}'
    )
    status = response.json()
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(2)

# 3. Get results
response = requests.get(
    f'http://localhost:8000/api/v1/analytics/{image_id}'
)
results = response.json()
print(results)
```

---

## WebSocket Support (Future)

Real-time updates via WebSocket will be added in future versions:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis/{image_id}');
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log('Progress:', update);
};
```

---

**Last Updated**: 2025-12-15  
**API Version**: 1.0.0

