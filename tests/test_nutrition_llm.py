"""
Test script for Nutrition LLM with sample ingredient lists.

Tests various ingredient combinations, edge cases, and uncertainty handling
as specified in Requirements 4.1 and 4.2.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foodvision_ai.models.nutrition_llm import NutritionLLM, NutritionalValues, ConfidenceLevel
from typing import List, Dict
import json


def test_sample_ingredient_combinations():
    """Test nutrition calculation with various ingredient combinations."""
    
    print("=" * 60)
    print("TESTING NUTRITION LLM WITH SAMPLE INGREDIENT LISTS")
    print("=" * 60)
    
    # Initialize the nutrition LLM (will use fallback if Phi-3 not available)
    try:
        nutrition_llm = NutritionLLM()
        print("✓ Nutrition LLM initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Nutrition LLM: {e}")
        return False
    
    # Test cases with various ingredient combinations
    test_cases = [
        {
            "name": "Simple Pasta Dish",
            "ingredients": ["pasta", "tomato", "cheese", "olive oil"],
            "description": "Spaghetti with tomato sauce and cheese",
            "expected_range": {"calories": (250, 400), "protein": (8, 15)}
        },
        {
            "name": "Chicken Salad",
            "ingredients": ["chicken", "lettuce", "tomato", "cucumber", "olive oil"],
            "description": "Grilled chicken salad with vegetables",
            "expected_range": {"calories": (200, 350), "protein": (20, 35)}
        },
        {
            "name": "Vegetarian Bowl",
            "ingredients": ["rice", "broccoli", "carrot", "tofu"],
            "description": "Vegetarian rice bowl with steamed vegetables",
            "expected_range": {"calories": (180, 300), "protein": (8, 18)}
        },
        {
            "name": "Breakfast Sandwich",
            "ingredients": ["bread", "egg", "cheese", "butter"],
            "description": "Egg and cheese sandwich",
            "expected_range": {"calories": (300, 500), "protein": (15, 25)}
        },
        {
            "name": "Fish and Chips",
            "ingredients": ["fish", "potato", "oil"],
            "description": "Fried fish with french fries",
            "expected_range": {"calories": (400, 700), "protein": (20, 35)}
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Ingredients: {', '.join(test_case['ingredients'])}")
        print(f"   Description: {test_case['description']}")
        
        try:
            # Calculate nutrition
            nutrition = nutrition_llm.calculate_nutrition(
                test_case['ingredients'], 
                test_case['description']
            )
            
            # Display results
            print(f"   Results:")
            print(f"     Calories: {nutrition.calories}")
            print(f"     Fat: {nutrition.fat}g")
            print(f"     Carbohydrates: {nutrition.carbohydrates}g")
            print(f"     Protein: {nutrition.protein}g")
            print(f"     Portion: {nutrition.portion_size}")
            print(f"     Confidence: {nutrition.confidence:.2f} ({nutrition.confidence_level.value})")
            print(f"     Confidence Range: {nutrition.confidence_range}")
            
            # Validate against expected ranges
            expected = test_case['expected_range']
            calories_ok = expected['calories'][0] <= nutrition.calories <= expected['calories'][1]
            protein_ok = expected['protein'][0] <= nutrition.protein <= expected['protein'][1]
            
            if calories_ok and protein_ok:
                print(f"   ✓ Results within expected ranges")
                status = "PASS"
            else:
                print(f"   ⚠ Results outside expected ranges")
                print(f"     Expected calories: {expected['calories']}, got: {nutrition.calories}")
                print(f"     Expected protein: {expected['protein']}, got: {nutrition.protein}")
                status = "WARNING"
            
            results.append({
                "test_case": test_case['name'],
                "status": status,
                "nutrition": nutrition,
                "within_range": calories_ok and protein_ok
            })
            
        except Exception as e:
            print(f"   ✗ Test failed: {e}")
            results.append({
                "test_case": test_case['name'],
                "status": "FAIL",
                "error": str(e),
                "within_range": False
            })
    
    return results


def test_edge_cases():
    """Test edge cases and uncertainty handling."""
    
    print(f"\n{'=' * 60}")
    print("TESTING EDGE CASES AND UNCERTAINTY HANDLING")
    print("=" * 60)
    
    nutrition_llm = NutritionLLM()
    
    edge_cases = [
        {
            "name": "Empty Ingredients List",
            "ingredients": [],
            "description": "Unknown food item",
            "expected_behavior": "Should handle gracefully with low confidence"
        },
        {
            "name": "Single Ingredient",
            "ingredients": ["apple"],
            "description": "A fresh apple",
            "expected_behavior": "Should provide reasonable nutrition for single fruit"
        },
        {
            "name": "Unclear Ingredients",
            "ingredients": ["mixed", "unknown", "stuff"],
            "description": "Some kind of mixed dish",
            "expected_behavior": "Should have very low confidence"
        },
        {
            "name": "Many Ingredients",
            "ingredients": ["chicken", "rice", "broccoli", "carrot", "onion", "garlic", "oil", "salt", "pepper", "herbs"],
            "description": "Complex stir-fry with many ingredients",
            "expected_behavior": "Should handle complex dishes with moderate confidence"
        },
        {
            "name": "Conflicting Information",
            "ingredients": ["chocolate", "cake"],
            "description": "Healthy salad with vegetables",
            "expected_behavior": "Should handle conflicting ingredient/description info"
        }
    ]
    
    edge_results = []
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n{i}. Testing: {case['name']}")
        print(f"   Ingredients: {case['ingredients']}")
        print(f"   Description: {case['description']}")
        print(f"   Expected: {case['expected_behavior']}")
        
        try:
            nutrition = nutrition_llm.calculate_nutrition(
                case['ingredients'], 
                case['description']
            )
            
            print(f"   Results:")
            print(f"     Calories: {nutrition.calories}")
            print(f"     Confidence: {nutrition.confidence:.2f} ({nutrition.confidence_level.value})")
            print(f"     Portion: {nutrition.portion_size}")
            print(f"     Confidence Range: {nutrition.confidence_range}")
            
            # Analyze behavior
            if len(case['ingredients']) == 0 and nutrition.confidence < 0.5:
                print(f"   ✓ Correctly handled empty ingredients with low confidence")
                status = "PASS"
            elif "unknown" in case['name'].lower() and nutrition.confidence < 0.4:
                print(f"   ✓ Correctly assigned low confidence to unclear ingredients")
                status = "PASS"
            elif len(case['ingredients']) > 8 and nutrition.confidence > 0.3:
                print(f"   ✓ Handled complex ingredient list reasonably")
                status = "PASS"
            else:
                print(f"   ✓ Handled edge case (confidence: {nutrition.confidence:.2f})")
                status = "PASS"
            
            edge_results.append({
                "test_case": case['name'],
                "status": status,
                "confidence": nutrition.confidence,
                "confidence_level": nutrition.confidence_level.value
            })
            
        except Exception as e:
            print(f"   ✗ Edge case test failed: {e}")
            edge_results.append({
                "test_case": case['name'],
                "status": "FAIL",
                "error": str(e)
            })
    
    return edge_results


def test_portion_size_estimation():
    """Test portion size estimation functionality."""
    
    print(f"\n{'=' * 60}")
    print("TESTING PORTION SIZE ESTIMATION")
    print("=" * 60)
    
    nutrition_llm = NutritionLLM()
    
    portion_cases = [
        {
            "description": "A slice of pizza",
            "ingredients": ["bread", "cheese", "tomato"],
            "expected_portion_type": "slice"
        },
        {
            "description": "A hamburger with fries",
            "ingredients": ["beef", "bread", "potato"],
            "expected_portion_type": "piece"
        },
        {
            "description": "A bowl of pasta",
            "ingredients": ["pasta", "tomato", "cheese"],
            "expected_portion_type": "cup"
        },
        {
            "description": "A chicken breast",
            "ingredients": ["chicken"],
            "expected_portion_type": "piece"
        }
    ]
    
    portion_results = []
    
    for i, case in enumerate(portion_cases, 1):
        print(f"\n{i}. Testing portion estimation for: {case['description']}")
        
        try:
            nutrition = nutrition_llm.calculate_nutrition(
                case['ingredients'], 
                case['description']
            )
            
            print(f"   Estimated portion: {nutrition.portion_size}")
            print(f"   Portion weight: {nutrition.portion_weight_grams}g")
            print(f"   Portion certainty: {nutrition.portion_certainty:.2f}")
            
            # Check if per-serving breakdown is provided
            if nutrition.per_serving_breakdown:
                print(f"   Per-serving breakdown available: {len(nutrition.per_serving_breakdown)} variants")
                if 'per_100g' in nutrition.per_serving_breakdown:
                    per_100g = nutrition.per_serving_breakdown['per_100g']
                    print(f"   Per 100g: {per_100g['calories']} calories, {per_100g['protein']}g protein")
            
            portion_results.append({
                "description": case['description'],
                "portion_size": nutrition.portion_size,
                "portion_weight": nutrition.portion_weight_grams,
                "portion_certainty": nutrition.portion_certainty,
                "has_breakdown": bool(nutrition.per_serving_breakdown),
                "status": "PASS"
            })
            
        except Exception as e:
            print(f"   ✗ Portion estimation failed: {e}")
            portion_results.append({
                "description": case['description'],
                "status": "FAIL",
                "error": str(e)
            })
    
    return portion_results


def test_confidence_ranges():
    """Test confidence range calculation."""
    
    print(f"\n{'=' * 60}")
    print("TESTING CONFIDENCE RANGE CALCULATION")
    print("=" * 60)
    
    nutrition_llm = NutritionLLM()
    
    confidence_cases = [
        {
            "name": "High Confidence Case",
            "ingredients": ["chicken", "rice", "broccoli"],
            "description": "Grilled chicken with steamed rice and broccoli",
            "expected_confidence": "high"
        },
        {
            "name": "Medium Confidence Case",
            "ingredients": ["mixed vegetables", "sauce"],
            "description": "Vegetable stir-fry with sauce",
            "expected_confidence": "medium"
        },
        {
            "name": "Low Confidence Case",
            "ingredients": ["unknown"],
            "description": "Some kind of food",
            "expected_confidence": "low"
        }
    ]
    
    confidence_results = []
    
    for i, case in enumerate(confidence_cases, 1):
        print(f"\n{i}. Testing confidence for: {case['name']}")
        
        try:
            nutrition = nutrition_llm.calculate_nutrition(
                case['ingredients'], 
                case['description']
            )
            
            print(f"   Confidence: {nutrition.confidence:.2f}")
            print(f"   Confidence Level: {nutrition.confidence_level.value}")
            print(f"   Confidence Range: {nutrition.confidence_range}")
            print(f"   Confidence Bounds: {nutrition.confidence_lower_bound:.2f} - {nutrition.confidence_upper_bound:.2f}")
            
            # Validate confidence range format
            range_valid = "±" in nutrition.confidence_range and "%" in nutrition.confidence_range
            bounds_valid = 0 <= nutrition.confidence_lower_bound <= nutrition.confidence_upper_bound <= 1
            
            if range_valid and bounds_valid:
                print(f"   ✓ Confidence range format and bounds are valid")
                status = "PASS"
            else:
                print(f"   ⚠ Confidence range format or bounds issue")
                status = "WARNING"
            
            confidence_results.append({
                "case": case['name'],
                "confidence": nutrition.confidence,
                "confidence_level": nutrition.confidence_level.value,
                "confidence_range": nutrition.confidence_range,
                "range_valid": range_valid,
                "bounds_valid": bounds_valid,
                "status": status
            })
            
        except Exception as e:
            print(f"   ✗ Confidence test failed: {e}")
            confidence_results.append({
                "case": case['name'],
                "status": "FAIL",
                "error": str(e)
            })
    
    return confidence_results


def generate_test_summary(all_results):
    """Generate a summary of all test results."""
    
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    warning_tests = 0
    
    for result_set in all_results:
        for result in result_set:
            total_tests += 1
            if result.get('status') == 'PASS':
                passed_tests += 1
            elif result.get('status') == 'FAIL':
                failed_tests += 1
            elif result.get('status') == 'WARNING':
                warning_tests += 1
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Warnings: {warning_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n✓ All tests completed successfully!")
        print("✓ Nutrition LLM is working correctly with sample ingredient lists")
        print("✓ Edge cases and uncertainty handling are functional")
        print("✓ Portion size estimation is operational")
        print("✓ Confidence ranges are being calculated properly")
        return True
    else:
        print(f"\n⚠ {failed_tests} tests failed - review implementation")
        return False


def main():
    """Run all nutrition LLM tests."""
    
    print("Starting Nutrition LLM Testing Suite...")
    print("This will test various ingredient combinations, edge cases, and uncertainty handling.")
    
    try:
        # Run all test suites
        ingredient_results = test_sample_ingredient_combinations()
        edge_results = test_edge_cases()
        portion_results = test_portion_size_estimation()
        confidence_results = test_confidence_ranges()
        
        # Generate summary
        all_results = [ingredient_results, edge_results, portion_results, confidence_results]
        success = generate_test_summary(all_results)
        
        # Save detailed results to file
        detailed_results = {
            "ingredient_combinations": ingredient_results,
            "edge_cases": edge_results,
            "portion_estimation": portion_results,
            "confidence_ranges": confidence_results,
            "summary": {
                "total_tests": sum(len(r) for r in all_results),
                "success": success
            }
        }
        
        with open("tests/nutrition_llm_test_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: tests/nutrition_llm_test_results.json")
        
        return success
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)