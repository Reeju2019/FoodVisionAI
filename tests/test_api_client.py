"""
Simple API client for testing the FoodVisionAI API
"""

import requests
import time
import json
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"

def test_health():
    """Test the health endpoint."""
    print("\n" + "=" * 70)
    print("🏥 Testing Health Endpoint")
    print("=" * 70)

    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("\nStart the server with:")
        print("  uvicorn foodvision_ai.api.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_upload(image_path: str):
    """Test uploading an image."""
    print("\n" + "=" * 70)
    print("📤 Testing Upload Endpoint")
    print("=" * 70)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"Uploading: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{API_BASE}/api/v1/upload",
                files=files,
                timeout=30
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            print(f"\n✅ Upload successful!")
            print(f"Image ID: {result['image_id']}")
            return result['image_id']
        else:
            print(f"❌ Upload failed!")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_status(image_id: str, max_polls: int = 120):
    """Test status endpoint and poll until complete."""
    print("\n" + "=" * 70)
    print("📊 Testing Status Endpoint")
    print("=" * 70)

    print(f"Image ID: {image_id}")
    print("Polling for status updates...")
    print("⏳ This may take 2-3 minutes (BLIP-2 is running on CPU)...\n")

    for i in range(max_polls):
        try:
            response = requests.get(
                f"{API_BASE}/api/v1/status/{image_id}",
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status', 'unknown')
                
                print(f"Poll {i+1}: Status = {status}")
                
                if status == 'completed':
                    print("\n✅ Analysis completed!")
                    print(f"\nFull Results:")
                    print(json.dumps(result, indent=2))
                    return result
                elif status == 'failed':
                    print("\n❌ Analysis failed!")
                    print(f"Results: {json.dumps(result, indent=2)}")
                    return result
                else:
                    # Still processing, wait and poll again
                    time.sleep(5)
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    print("\n⏱️  Timeout: Analysis took too long")
    return None


def download_test_image():
    """Download a test image if none exists."""
    test_image = "test_salad.jpg"
    
    if Path(test_image).exists():
        print(f"✅ Test image already exists: {test_image}")
        return test_image
    
    print(f"📥 Downloading test image...")
    
    try:
        import urllib.request
        url = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400"
        urllib.request.urlretrieve(url, test_image)
        print(f"✅ Downloaded: {test_image}")
        return test_image
    except Exception as e:
        print(f"❌ Failed to download test image: {e}")
        return None


def main():
    """Run all API tests."""
    print("=" * 70)
    print("🧪 FoodVisionAI API Test Client")
    print("=" * 70)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ API is not running. Start it first!")
        return
    
    # Test 2: Download test image
    test_image = download_test_image()
    if not test_image:
        print("\n❌ No test image available")
        return
    
    # Test 3: Upload image
    image_id = test_upload(test_image)
    if not image_id:
        print("\n❌ Upload failed")
        return
    
    # Test 4: Poll status
    result = test_status(image_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    if result and result.get('status') == 'completed':
        print("✅ ALL TESTS PASSED!")
        print(f"\nView results in browser:")
        print(f"  {API_BASE}/api/v1/analytics/{image_id}")
    else:
        print("❌ Some tests failed. Check errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

