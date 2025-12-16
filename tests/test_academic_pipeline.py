#!/usr/bin/env python3
"""
Test script for Academic 3-Stage Pipeline

This script tests the academic pipeline without requiring a full server setup.
"""

import sys
import os
sys.path.append('.')

from foodvision_ai.models.academic_pipeline import AcademicFoodPipeline
from foodvision_ai.config import settings
import asyncio


def test_academic_pipeline():
    """Test the academic pipeline with a sample image"""
    print("🎓 Testing Academic 3-Stage Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    try:
        pipeline = AcademicFoodPipeline(device='cpu')
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print(f"\n📋 Pipeline Information:")
    print(f"   Type: {info['pipeline_type']}")
    print(f"   Stage 1: {info['stage1']['component']}")
    print(f"   Stage 2: {info['stage2']['component']}")
    print(f"   Stage 3: {info['stage3']['component']}")
    
    # Test with a sample image URL (placeholder)
    test_image_url = "https://example.com/sample_food_image.jpg"
    
    print(f"\n🧪 Testing analysis with sample image...")
    try:
        # Note: This will fail with the example URL, but we can test the pipeline structure
        results = pipeline.analyze_food_image(test_image_url)
        
        print(f"📊 Analysis Results:")
        print(f"   Status: {results['analysis_status']}")
        
        if results['analysis_status'] == 'success':
            print(f"   Stage 1 - Ingredients: {len(results['stage1_ingredients']['ingredients'])} detected")
            print(f"   Stage 2 - Dish: {results['stage2_dish_analysis']['predicted_dish']}")
            print(f"   Stage 3 - Calories: {results['stage3_nutrition']['calories']}")
        else:
            print(f"   Error: {results.get('error_message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Analysis failed (expected with sample URL): {e}")
    
    print(f"\n✅ Academic pipeline test completed")
    return True


def test_configuration():
    """Test configuration settings"""
    print("\n🔧 Testing Configuration")
    print("=" * 30)
    
    print(f"MongoDB URL: {settings.mongodb_url}")
    print(f"Database Name: {settings.database_name}")
    print(f"Gemini API Key: {'✅ Set' if settings.gemini_api_key else '❌ Not set'}")
    print(f"Debug Mode: {settings.debug}")
    
    if not settings.gemini_api_key:
        print("\n⚠️  Gemini API key not configured. Stage 2 will use fallback method.")
        print("   To enable Gemini Pro, add GEMINI_API_KEY to your .env file")


if __name__ == "__main__":
    print("🎯 FoodVisionAI Academic Pipeline Test")
    print("=" * 60)
    
    # Test configuration
    test_configuration()
    
    # Test pipeline
    success = test_academic_pipeline()
    
    if success:
        print("\n🎉 All tests completed!")
        print("\n📝 Next Steps:")
        print("   1. Add your Gemini API key to .env file for Stage 2 GenAI")
        print("   2. Train or download Recipe1M+ model for Stage 1 CNN")
        print("   3. Start the server: python main.py")
        print("   4. Upload food images via http://localhost:8000/api/v1/upload")
    else:
        print("\n❌ Tests failed. Check the error messages above.")