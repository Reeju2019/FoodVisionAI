"""
Test script for Vision Model with sample Google Drive public links

Tests the vision model functionality with various food images from public URLs
to validate ingredient extraction and description generation.
"""

import sys
import os
import asyncio
from typing import List, Dict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foodvision_ai.models.vision_model import VisionModel, analyze_food_image
from foodvision_ai.utils.logging_config import setup_logging


def test_sample_food_images():
    """Test Vision Model with sample food images from public URLs."""
    
    # Setup logging
    setup_logging()
    
    # Sample food images (using placeholder URLs - replace with actual public Google Drive links)
    sample_images = [
        {
            "name": "Pizza",
            "url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=500",
            "expected_ingredients": ["cheese", "tomato", "flour"],
            "expected_food_type": "pizza"
        },
        {
            "name": "Burger",
            "url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500",
            "expected_ingredients": ["beef", "cheese", "lettuce"],
            "expected_food_type": "hamburger"
        },
        {
            "name": "Salad",
            "url": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=500",
            "expected_ingredients": ["lettuce", "tomato", "vegetables"],
            "expected_food_type": "salad"
        },
        {
            "name": "Pasta",
            "url": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=500",
            "expected_ingredients": ["pasta", "tomato", "cheese"],
            "expected_food_type": "pasta"
        },
        {
            "name": "Sushi",
            "url": "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=500",
            "expected_ingredients": ["fish", "rice"],
            "expected_food_type": "sushi"
        }
    ]
    
    print("=" * 80)
    print("VISION MODEL TEST - Sample Food Images")
    print("=" * 80)
    
    # Initialize Vision Model
    try:
        print("Initializing Vision Model...")
        vision_model = VisionModel()
        print("✓ Vision Model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Vision Model: {e}")
        return False
    
    # Test results
    test_results = []
    successful_tests = 0
    
    for i, sample in enumerate(sample_images, 1):
        print(f"\n{'-' * 60}")
        print(f"Test {i}: {sample['name']}")
        print(f"URL: {sample['url']}")
        print(f"{'-' * 60}")
        
        try:
            # Analyze the image
            result = vision_model.analyze_image(sample['url'])
            
            if result['analysis_status'] == 'success':
                print("✓ Analysis completed successfully")
                
                # Display results
                print(f"Food Type: {result['food_type']}")
                print(f"Description: {result['description']}")
                print(f"Confidence: {result['confidence']:.3f} ({result['confidence_level']})")
                print(f"Ingredients: {', '.join(result['ingredients'])}")
                print(f"Multiple Foods Detected: {result['multiple_foods']}")
                
                # Multi-food analysis
                if result['multiple_foods']:
                    print("\nMulti-food Analysis:")
                    for food_item in result['multi_food_analysis']['food_items']:
                        print(f"  - {food_item['food_type']}: {food_item['confidence']:.3f} ({food_item['likelihood']})")
                
                # Top predictions
                print("\nTop Predictions:")
                for j, pred in enumerate(result['top_predictions'][:3], 1):
                    print(f"  {j}. {pred['food_type']}: {pred['confidence']:.3f}")
                
                # Validation checks
                validation_passed = True
                validation_messages = []
                
                # Check if any expected ingredients are found
                found_ingredients = [ing.lower() for ing in result['ingredients']]
                expected_found = any(exp.lower() in found_ingredients for exp in sample['expected_ingredients'])
                if not expected_found:
                    validation_passed = False
                    validation_messages.append(f"Expected ingredients not found: {sample['expected_ingredients']}")
                
                # Check confidence level
                if result['confidence'] < 0.1:
                    validation_passed = False
                    validation_messages.append("Very low confidence score")
                
                # Check if description is meaningful
                if len(result['description']) < 5:
                    validation_passed = False
                    validation_messages.append("Description too short")
                
                if validation_passed:
                    print("✓ Validation: PASSED")
                    successful_tests += 1
                else:
                    print("⚠ Validation: PARTIAL")
                    for msg in validation_messages:
                        print(f"  - {msg}")
                
                test_results.append({
                    'name': sample['name'],
                    'status': 'success',
                    'validation_passed': validation_passed,
                    'result': result
                })
                
            else:
                print(f"✗ Analysis failed: {result.get('error_message', 'Unknown error')}")
                test_results.append({
                    'name': sample['name'],
                    'status': 'failed',
                    'validation_passed': False,
                    'error': result.get('error_message', 'Unknown error')
                })
        
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            test_results.append({
                'name': sample['name'],
                'status': 'exception',
                'validation_passed': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Tests: {len(sample_images)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(sample_images) - successful_tests}")
    print(f"Success Rate: {(successful_tests / len(sample_images)) * 100:.1f}%")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for result in test_results:
        status_icon = "✓" if result['validation_passed'] else "✗"
        print(f"  {status_icon} {result['name']}: {result['status']}")
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    return successful_tests >= len(sample_images) * 0.6  # 60% success rate threshold


def test_edge_cases():
    """Test Vision Model with edge cases and various image qualities."""
    
    print(f"\n{'=' * 80}")
    print("VISION MODEL TEST - Edge Cases")
    print(f"{'=' * 80}")
    
    # Edge case URLs (these should be replaced with actual test images)
    edge_cases = [
        {
            "name": "Low Quality Image",
            "url": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=100",  # Very small image
            "description": "Testing with low resolution image"
        },
        {
            "name": "Multiple Foods",
            "url": "https://images.unsplash.com/photo-1555939594-58d7cb561ad1?w=500",  # Breakfast plate
            "description": "Testing multi-food detection"
        },
        {
            "name": "Unclear Food",
            "url": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=500",  # Abstract food
            "description": "Testing with unclear/artistic food image"
        }
    ]
    
    try:
        vision_model = VisionModel()
        edge_case_results = []
        
        for i, case in enumerate(edge_cases, 1):
            print(f"\nEdge Case {i}: {case['name']}")
            print(f"Description: {case['description']}")
            print(f"URL: {case['url']}")
            
            try:
                result = vision_model.analyze_image(case['url'])
                
                if result['analysis_status'] == 'success':
                    print(f"✓ Analysis completed")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Multiple Foods: {result['multiple_foods']}")
                    print(f"  Ingredients Count: {len(result['ingredients'])}")
                    
                    edge_case_results.append({
                        'name': case['name'],
                        'success': True,
                        'confidence': result['confidence']
                    })
                else:
                    print(f"✗ Analysis failed: {result.get('error_message', 'Unknown')}")
                    edge_case_results.append({
                        'name': case['name'],
                        'success': False,
                        'error': result.get('error_message', 'Unknown')
                    })
                    
            except Exception as e:
                print(f"✗ Exception: {e}")
                edge_case_results.append({
                    'name': case['name'],
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\nEdge Case Summary:")
        for result in edge_case_results:
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {result['name']}")
            if not result['success']:
                print(f"    Error: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")
        return False


def test_convenience_function():
    """Test the convenience function for quick analysis."""
    
    print(f"\n{'=' * 80}")
    print("VISION MODEL TEST - Convenience Function")
    print(f"{'=' * 80}")
    
    test_url = "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=500"  # Pizza image
    
    try:
        print("Testing analyze_food_image convenience function...")
        result = analyze_food_image(test_url)
        
        if result['analysis_status'] == 'success':
            print("✓ Convenience function works correctly")
            print(f"  Food Type: {result['food_type']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            return True
        else:
            print(f"✗ Convenience function failed: {result.get('error_message', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        return False


def main():
    """Run all vision model tests."""
    
    print("Starting Vision Model Tests...")
    print("Note: This test uses public Unsplash images as placeholders.")
    print("For production testing, replace with actual Google Drive public links.")
    
    # Run all tests
    test1_passed = test_sample_food_images()
    test2_passed = test_edge_cases()
    test3_passed = test_convenience_function()
    
    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL TEST SUMMARY")
    print(f"{'=' * 80}")
    
    tests = [
        ("Sample Food Images", test1_passed),
        ("Edge Cases", test2_passed),
        ("Convenience Function", test3_passed)
    ]
    
    passed_tests = sum(1 for _, passed in tests if passed)
    total_tests = len(tests)
    
    for test_name, passed in tests:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Vision Model is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)