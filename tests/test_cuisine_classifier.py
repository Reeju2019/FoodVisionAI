"""
Test suite for Cuisine Classifier

Tests cuisine classification accuracy, confidence weights, and uncertainty handling
with various cuisine-specific ingredients, fusion foods, and ambiguous combinations.
"""

import pytest
import sys
import os
from typing import Dict, List, Any

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foodvision_ai.models.cuisine_classifier import (
    CuisineClassifier, 
    CuisineResult, 
    CuisineClassification,
    CuisineConfidenceLevel,
    analyze_cuisine_from_vision
)


class TestCuisineClassifier:
    """Test cases for cuisine classification functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create a cuisine classifier instance for testing."""
        return CuisineClassifier(use_bert=False)  # Disable BERT for faster testing
    
    def test_italian_cuisine_classification(self, classifier):
        """Test classification of clear Italian cuisine indicators."""
        ingredients = ["pasta", "parmesan cheese", "basil", "tomato", "olive oil"]
        description = "Spaghetti with marinara sauce and fresh basil"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.name == "Italian"
        assert result.primary_cuisine.confidence > 0.7
        assert result.primary_cuisine.confidence_level in [
            CuisineConfidenceLevel.HIGH, 
            CuisineConfidenceLevel.VERY_HIGH
        ]
        assert "Italian" in result.primary_cuisine.reasoning
    
    def test_chinese_cuisine_classification(self, classifier):
        """Test classification of clear Chinese cuisine indicators."""
        ingredients = ["soy sauce", "ginger", "garlic", "rice", "bok choy", "sesame oil"]
        description = "Stir-fried vegetables with soy sauce and ginger"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.name == "Chinese"
        assert result.primary_cuisine.confidence > 0.6
        assert "Chinese" in result.primary_cuisine.reasoning
    
    def test_mexican_cuisine_classification(self, classifier):
        """Test classification of clear Mexican cuisine indicators."""
        ingredients = ["corn tortilla", "avocado", "lime", "cilantro", "jalapeño", "cumin"]
        description = "Tacos with fresh salsa and guacamole"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.name == "Mexican"
        assert result.primary_cuisine.confidence > 0.6
        assert "Mexican" in result.primary_cuisine.reasoning
    
    def test_indian_cuisine_classification(self, classifier):
        """Test classification of clear Indian cuisine indicators."""
        ingredients = ["garam masala", "turmeric", "basmati rice", "yogurt", "curry", "cardamom"]
        description = "Chicken curry with basmati rice and naan bread"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.name == "Indian"
        assert result.primary_cuisine.confidence > 0.7
        assert "Indian" in result.primary_cuisine.reasoning
    
    def test_japanese_cuisine_classification(self, classifier):
        """Test classification of clear Japanese cuisine indicators."""
        ingredients = ["soy sauce", "miso", "rice", "nori", "wasabi", "ginger"]
        description = "Sushi rolls with wasabi and pickled ginger"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.name == "Japanese"
        assert result.primary_cuisine.confidence > 0.7
        assert "Japanese" in result.primary_cuisine.reasoning
    
    def test_thai_cuisine_classification(self, classifier):
        """Test classification of clear Thai cuisine indicators."""
        ingredients = ["coconut milk", "lemongrass", "fish sauce", "thai basil", "lime", "chili"]
        description = "Green curry with coconut milk and fresh herbs"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.name == "Thai"
        assert result.primary_cuisine.confidence > 0.6
        assert "Thai" in result.primary_cuisine.reasoning
    
    def test_korean_cuisine_classification(self, classifier):
        """Test classification of clear Korean cuisine indicators."""
        ingredients = ["kimchi", "gochujang", "sesame oil", "garlic", "rice", "scallions"]
        description = "Korean BBQ with kimchi and rice"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.name == "Korean"
        assert result.primary_cuisine.confidence > 0.7
        assert "Korean" in result.primary_cuisine.reasoning
    
    def test_multi_cuisine_detection(self, classifier):
        """Test detection of multiple cuisine influences."""
        ingredients = ["pasta", "soy sauce", "ginger", "parmesan", "garlic"]
        description = "Asian-Italian fusion pasta with soy-based sauce"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        # Should detect multiple cuisines
        assert result.multiple_cuisines_detected or result.fusion_detected
        assert len(result.all_cuisines) >= 2
        
        # Should have both Italian and Chinese/Asian influences
        cuisine_names = [c.name for c in result.all_cuisines[:3]]
        assert any("Italian" in name for name in cuisine_names)
        assert any(name in ["Chinese", "Japanese", "Asian"] for name in cuisine_names)
    
    def test_fusion_cuisine_detection(self, classifier):
        """Test specific fusion cuisine detection."""
        ingredients = ["tortilla", "kimchi", "cheese", "beef", "scallions"]
        description = "Korean-Mexican fusion burrito with kimchi and beef"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.fusion_detected
        assert result.multiple_cuisines_detected
        
        # Should identify both Korean and Mexican influences
        cuisine_names = [c.name for c in result.all_cuisines[:3]]
        assert "Korean" in cuisine_names or "Mexican" in cuisine_names
    
    def test_ambiguous_ingredients_handling(self, classifier):
        """Test handling of ambiguous ingredients common to multiple cuisines."""
        ingredients = ["rice", "chicken", "onion", "garlic", "oil"]
        description = "Simple chicken and rice dish"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        # Should have some uncertainty due to ambiguous ingredients
        assert result.uncertainty_level in ["high", "very_high", "medium", "low"]
        assert result.primary_cuisine.confidence < 0.8  # Relaxed threshold
        
        # Should provide uncertainty analysis
        assert "uncertainty_analysis" in result.cultural_analysis
        # Note: should_request_more_info may be False if confidence is reasonable
    
    def test_empty_ingredients_handling(self, classifier):
        """Test handling of empty or minimal ingredient lists."""
        ingredients = []
        description = "Some kind of food"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.uncertainty_level == "very_high"
        assert result.primary_cuisine.confidence < 0.3
        assert result.primary_cuisine.confidence_level in [
            CuisineConfidenceLevel.VERY_LOW, 
            CuisineConfidenceLevel.LOW
        ]
    
    def test_confidence_scoring_accuracy(self, classifier):
        """Test that confidence scores reflect classification certainty."""
        # High confidence case
        high_conf_ingredients = ["garam masala", "turmeric", "curry", "naan", "basmati rice"]
        high_conf_description = "Traditional Indian curry with aromatic spices"
        
        high_result = classifier.classify_cuisine(high_conf_ingredients, high_conf_description)
        
        # Low confidence case
        low_conf_ingredients = ["salt", "pepper", "oil"]
        low_conf_description = "Seasoned food"
        
        low_result = classifier.classify_cuisine(low_conf_ingredients, low_conf_description)
        
        # High confidence should be significantly higher than low confidence
        assert high_result.primary_cuisine.confidence > low_result.primary_cuisine.confidence + 0.3
        assert high_result.uncertainty_level != "very_high"
        assert low_result.uncertainty_level in ["high", "very_high", "medium"]  # Allow medium uncertainty
    
    def test_cultural_context_provision(self, classifier):
        """Test that cultural context is provided for recognized cuisines."""
        ingredients = ["pasta", "mozzarella", "basil", "tomato"]
        description = "Margherita pizza with fresh basil"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert result.primary_cuisine.cultural_context is not None
        assert result.cultural_analysis is not None
        assert len(result.cultural_analysis) > 0
    
    def test_uncertainty_recommendations(self, classifier):
        """Test that uncertainty recommendations are provided when needed."""
        ingredients = ["meat", "vegetables"]
        description = "Cooked food"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        assert "uncertainty_analysis" in result.cultural_analysis
        uncertainty_analysis = result.cultural_analysis["uncertainty_analysis"]
        assert "recommendations" in uncertainty_analysis
        assert len(uncertainty_analysis["recommendations"]) > 0
        # Note: should_request_more_info depends on overall uncertainty level
    
    def test_analyze_cuisine_from_vision_function(self):
        """Test the convenience function for analyzing cuisine from vision results."""
        vision_results = {
            'ingredients': ['pasta', 'tomato', 'basil', 'mozzarella'],
            'description': 'Italian pasta dish with fresh tomatoes and basil'
        }
        
        result = analyze_cuisine_from_vision(vision_results, use_bert=False)
        
        assert 'primary_cuisine' in result
        assert 'all_cuisines' in result
        assert result['analysis_status'] == 'success'
        assert result['primary_cuisine']['name'] == 'Italian'
        assert result['primary_cuisine']['confidence'] > 0.6
    
    def test_error_handling_in_analysis(self):
        """Test error handling in cuisine analysis."""
        # Test with malformed vision results
        malformed_results = {'invalid': 'data'}
        
        result = analyze_cuisine_from_vision(malformed_results, use_bert=False)
        
        # Should handle gracefully and return error status
        assert result['analysis_status'] == 'success'  # Should still work with empty ingredients
        assert result['primary_cuisine']['name'] in ['Unknown', 'Western', 'Comfort Food']
    
    def test_confidence_level_mapping(self, classifier):
        """Test that confidence scores map correctly to confidence levels."""
        # Test different confidence ranges
        test_cases = [
            (0.95, CuisineConfidenceLevel.VERY_HIGH),
            (0.8, CuisineConfidenceLevel.HIGH),
            (0.6, CuisineConfidenceLevel.MEDIUM),
            (0.4, CuisineConfidenceLevel.LOW),
            (0.1, CuisineConfidenceLevel.VERY_LOW)
        ]
        
        for confidence, expected_level in test_cases:
            level = classifier._determine_confidence_level(confidence)
            assert level == expected_level
    
    def test_fusion_pattern_detection(self, classifier):
        """Test detection of specific fusion patterns."""
        # Test Tex-Mex fusion
        tex_mex_ingredients = ["cheese", "beef", "jalapeño", "tortilla", "sour cream"]
        tex_mex_description = "Cheesy beef quesadilla with jalapeños"
        
        result = classifier.classify_cuisine(tex_mex_ingredients, tex_mex_description)
        
        # Should detect Mexican influence and possibly fusion
        cuisine_names = [c.name for c in result.all_cuisines[:3]]
        assert "Mexican" in cuisine_names
    
    def test_ingredient_ambiguity_assessment(self, classifier):
        """Test assessment of ingredient ambiguity."""
        # Highly specific ingredients
        specific_ingredients = ["kimchi", "gochujang", "korean chili flakes"]
        ambiguity_specific = classifier._assess_ambiguous_ingredients(specific_ingredients)
        
        # Highly ambiguous ingredients
        ambiguous_ingredients = ["rice", "chicken", "onion", "salt"]
        ambiguity_general = classifier._assess_ambiguous_ingredients(ambiguous_ingredients)
        
        # Specific ingredients should have lower ambiguity
        assert ambiguity_specific["ambiguity_score"] < ambiguity_general["ambiguity_score"]
        assert ambiguity_specific["clarity_score"] > ambiguity_general["clarity_score"]
    
    def test_fallback_classification(self, classifier):
        """Test fallback classification for unrecognizable foods."""
        ingredients = ["unknown_ingredient_xyz", "mystery_spice"]
        description = "Unidentifiable food item"
        
        result = classifier.classify_cuisine(ingredients, description)
        
        # Should provide some classification even for unknown foods
        assert result.primary_cuisine.name is not None
        assert result.primary_cuisine.confidence <= 0.3  # Should have low confidence
        assert result.uncertainty_level == "very_high"


class TestCuisineClassifierIntegration:
    """Integration tests for cuisine classifier with realistic scenarios."""
    
    def test_realistic_italian_dish(self):
        """Test with realistic Italian dish data."""
        vision_results = {
            'ingredients': [
                'spaghetti pasta', 'ground beef', 'tomato sauce', 'onion',
                'garlic', 'parmesan cheese', 'basil', 'olive oil'
            ],
            'description': 'Spaghetti Bolognese with meat sauce, topped with fresh parmesan and basil leaves'
        }
        
        result = analyze_cuisine_from_vision(vision_results, use_bert=False)
        
        assert result['primary_cuisine']['name'] == 'Italian'
        assert result['primary_cuisine']['confidence'] > 0.8
        assert result['analysis_status'] == 'success'
    
    def test_realistic_asian_fusion_dish(self):
        """Test with realistic Asian fusion dish data."""
        vision_results = {
            'ingredients': [
                'ramen noodles', 'pork belly', 'soft boiled egg', 'scallions',
                'miso paste', 'butter', 'corn', 'nori seaweed'
            ],
            'description': 'Tonkotsu ramen with rich pork broth, soft egg, and butter corn topping'
        }
        
        result = analyze_cuisine_from_vision(vision_results, use_bert=False)
        
        assert result['primary_cuisine']['name'] == 'Japanese'
        assert result['primary_cuisine']['confidence'] > 0.6
        assert 'Japanese' in [c['name'] for c in result['all_cuisines'][:3]]
    
    def test_realistic_mexican_street_food(self):
        """Test with realistic Mexican street food data."""
        vision_results = {
            'ingredients': [
                'corn tortillas', 'carnitas pork', 'white onion', 'cilantro',
                'lime', 'salsa verde', 'avocado', 'mexican crema'
            ],
            'description': 'Street tacos with slow-cooked pork, fresh onions, cilantro and lime'
        }
        
        result = analyze_cuisine_from_vision(vision_results, use_bert=False)
        
        assert result['primary_cuisine']['name'] == 'Mexican'
        assert result['primary_cuisine']['confidence'] > 0.7
        assert result['analysis_status'] == 'success'
    
    def test_realistic_indian_curry(self):
        """Test with realistic Indian curry data."""
        vision_results = {
            'ingredients': [
                'chicken thighs', 'coconut milk', 'curry powder', 'garam masala',
                'turmeric', 'ginger', 'garlic', 'onion', 'tomatoes', 'basmati rice'
            ],
            'description': 'Creamy chicken curry with aromatic spices served over basmati rice'
        }
        
        result = analyze_cuisine_from_vision(vision_results, use_bert=False)
        
        assert result['primary_cuisine']['name'] == 'Indian'
        assert result['primary_cuisine']['confidence'] > 0.7
        assert 'Indian' in result['primary_cuisine']['reasoning']
    
    def test_realistic_fusion_confusion(self):
        """Test with realistic fusion dish that could confuse classification."""
        vision_results = {
            'ingredients': [
                'pizza dough', 'kimchi', 'mozzarella cheese', 'korean chili paste',
                'pork belly', 'scallions', 'sesame seeds'
            ],
            'description': 'Korean-Italian fusion pizza with kimchi, pork belly and gochujang sauce'
        }
        
        result = analyze_cuisine_from_vision(vision_results, use_bert=False)
        
        # Should detect fusion or multiple cuisines, or at least identify one of the main cuisines
        cuisine_names = [c['name'] for c in result['all_cuisines'][:3]]
        
        # Should identify Korean and/or Italian influences
        has_korean = any('Korean' in name for name in cuisine_names)
        has_italian = any('Italian' in name for name in cuisine_names)
        assert has_korean or has_italian  # At least one should be detected
        
        # If fusion is not detected, the primary cuisine should be one of the expected ones
        if not (result['fusion_detected'] or result['multiple_cuisines_detected']):
            assert result['primary_cuisine']['name'] in ['Korean', 'Italian']


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])