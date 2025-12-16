"""
AI Models package for FoodVisionAI

Contains the three core AI models:
- Vision Model: Food recognition and ingredient extraction
- Nutrition LLM: Nutritional value calculation
- Cuisine Classifier: Cuisine type identification
"""

from .vision_model import VisionModel, analyze_food_image
from .nutrition_llm import NutritionLLM, NutritionalValues, ConfidenceLevel, analyze_nutrition_from_vision
from .cuisine_classifier import CuisineClassifier, CuisineResult, CuisineClassification, CuisineConfidenceLevel, analyze_cuisine_from_vision

__all__ = [
    'VisionModel', 'analyze_food_image',
    'NutritionLLM', 'NutritionalValues', 'ConfidenceLevel', 'analyze_nutrition_from_vision',
    'CuisineClassifier', 'CuisineResult', 'CuisineClassification', 'CuisineConfidenceLevel', 'analyze_cuisine_from_vision'
]