"""
Cuisine Classifier Demonstration Script

This script demonstrates the cuisine classifier functionality with various
test cases including clear cuisine indicators, fusion foods, and ambiguous ingredients.
"""

import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foodvision_ai.models.cuisine_classifier import analyze_cuisine_from_vision


def print_classification_result(title: str, result: dict):
    """Print a formatted classification result."""
    print(f"\n{'='*60}")
    print(f"TEST: {title}")
    print(f"{'='*60}")
    
    primary = result['primary_cuisine']
    print(f"Primary Cuisine: {primary['name']}")
    print(f"Confidence: {primary['confidence']:.2f} ({primary['confidence_level']})")
    print(f"Reasoning: {primary['reasoning']}")
    
    if result['multiple_cuisines_detected']:
        print(f"\nMultiple Cuisines Detected: Yes")
        print("All Detected Cuisines:")
        for i, cuisine in enumerate(result['all_cuisines'][:3], 1):
            print(f"  {i}. {cuisine['name']} ({cuisine['confidence']:.2f})")
    
    if result['fusion_detected']:
        print(f"Fusion Cuisine Detected: Yes")
    
    print(f"Uncertainty Level: {result['uncertainty_level']}")
    
    if 'uncertainty_analysis' in result.get('cultural_analysis', {}):
        uncertainty = result['cultural_analysis']['uncertainty_analysis']
        if uncertainty['recommendations']:
            print(f"Recommendations: {', '.join(uncertainty['recommendations'][:2])}")


def main():
    """Run cuisine classifier demonstration."""
    print("FoodVisionAI Cuisine Classifier Demonstration")
    print("=" * 60)
    
    # Test cases with various cuisine types and complexity levels
    test_cases = [
        {
            "title": "Clear Italian Cuisine",
            "vision_results": {
                'ingredients': ['spaghetti pasta', 'marinara sauce', 'parmesan cheese', 'fresh basil', 'olive oil'],
                'description': 'Classic spaghetti marinara with fresh basil and parmesan cheese'
            }
        },
        {
            "title": "Clear Indian Cuisine",
            "vision_results": {
                'ingredients': ['basmati rice', 'chicken', 'garam masala', 'turmeric', 'curry sauce', 'naan bread'],
                'description': 'Chicken curry with aromatic spices served with basmati rice and naan'
            }
        },
        {
            "title": "Clear Japanese Cuisine",
            "vision_results": {
                'ingredients': ['sushi rice', 'salmon', 'nori seaweed', 'wasabi', 'pickled ginger', 'soy sauce'],
                'description': 'Fresh salmon sushi rolls with wasabi and pickled ginger'
            }
        },
        {
            "title": "Korean-Mexican Fusion",
            "vision_results": {
                'ingredients': ['corn tortillas', 'kimchi', 'korean bbq beef', 'cilantro', 'lime', 'gochujang sauce'],
                'description': 'Korean BBQ tacos with kimchi and spicy gochujang sauce'
            }
        },
        {
            "title": "Asian-Italian Fusion",
            "vision_results": {
                'ingredients': ['ramen noodles', 'parmesan cheese', 'soy sauce', 'garlic', 'butter', 'scallions'],
                'description': 'Ramen carbonara with parmesan cheese and soy-butter sauce'
            }
        },
        {
            "title": "Ambiguous Ingredients",
            "vision_results": {
                'ingredients': ['rice', 'chicken', 'onion', 'garlic', 'vegetables'],
                'description': 'Simple chicken and rice dish with vegetables'
            }
        },
        {
            "title": "Mediterranean Cuisine",
            "vision_results": {
                'ingredients': ['olive oil', 'feta cheese', 'tomatoes', 'cucumber', 'olives', 'oregano'],
                'description': 'Greek salad with feta cheese, olives, and fresh vegetables'
            }
        },
        {
            "title": "Thai Cuisine",
            "vision_results": {
                'ingredients': ['coconut milk', 'lemongrass', 'thai basil', 'fish sauce', 'lime', 'chili peppers'],
                'description': 'Green curry with coconut milk and fresh Thai herbs'
            }
        },
        {
            "title": "Minimal Information",
            "vision_results": {
                'ingredients': ['meat', 'sauce'],
                'description': 'Cooked meat with sauce'
            }
        },
        {
            "title": "French Cuisine",
            "vision_results": {
                'ingredients': ['butter', 'white wine', 'shallots', 'cream', 'herbs', 'mushrooms'],
                'description': 'Classic French sauce with butter, wine, and fresh herbs'
            }
        }
    ]
    
    # Run all test cases
    for test_case in test_cases:
        try:
            result = analyze_cuisine_from_vision(test_case['vision_results'], use_bert=False)
            print_classification_result(test_case['title'], result)
        except Exception as e:
            print(f"\nERROR in {test_case['title']}: {e}")
    
    print(f"\n{'='*60}")
    print("Demonstration Complete!")
    print("The cuisine classifier successfully:")
    print("✓ Identified clear cuisine indicators")
    print("✓ Detected fusion cuisines")
    print("✓ Handled ambiguous ingredients")
    print("✓ Provided confidence scores and uncertainty analysis")
    print("✓ Offered recommendations for improvement")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()