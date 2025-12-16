"""
Simplified Nutrition LLM for FoodVisionAI

Implements nutritional analysis with fallback calculations for testing.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for nutritional calculations."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class NutritionalValues:
    """Structured nutritional values with confidence metrics."""
    calories: float
    fat: float  # grams
    carbohydrates: float  # grams
    protein: float  # grams
    fiber: Optional[float] = None  # grams
    sugar: Optional[float] = None  # grams
    sodium: Optional[float] = None  # mg
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    portion_size: str = "1 serving"
    portion_weight_grams: Optional[float] = None
    portion_certainty: float = 0.5
    confidence_range: str = "±15%"
    confidence_lower_bound: float = 0.0
    confidence_upper_bound: float = 1.0
    per_serving_breakdown: Optional[Dict[str, Dict[str, float]]] = None


class NutritionLLM:
    """
    Simplified Nutrition LLM for calculating nutritional values from food descriptions.
    
    Uses fallback database calculations for reliable testing.
    """
    
    def __init__(self, model_name: str = "fallback", device: Optional[str] = None):
        """
        Initialize the Nutrition LLM.
        
        Args:
            model_name: Model name (using fallback for testing)
            device: Device to run on (not used in fallback mode)
        """
        self.model_name = model_name
        self.device = device or 'cpu'
        
        # Load nutrition database
        self.nutrition_database = self._load_nutrition_database()
        
        print(f"Initialized Nutrition LLM with fallback calculations")
    
    def _load_nutrition_database(self) -> Dict[str, Dict[str, float]]:
        """Load basic nutritional database for common foods."""
        return {
            # Proteins
            "chicken": {"calories": 165, "fat": 3.6, "carbohydrates": 0, "protein": 31, "fiber": 0},
            "beef": {"calories": 250, "fat": 15, "carbohydrates": 0, "protein": 26, "fiber": 0},
            "fish": {"calories": 206, "fat": 12, "carbohydrates": 0, "protein": 22, "fiber": 0},
            "salmon": {"calories": 208, "fat": 13, "carbohydrates": 0, "protein": 20, "fiber": 0},
            "egg": {"calories": 155, "fat": 11, "carbohydrates": 1.1, "protein": 13, "fiber": 0},
            
            # Carbohydrates
            "rice": {"calories": 130, "fat": 0.3, "carbohydrates": 28, "protein": 2.7, "fiber": 0.4},
            "pasta": {"calories": 131, "fat": 1.1, "carbohydrates": 25, "protein": 5, "fiber": 1.8},
            "bread": {"calories": 265, "fat": 3.2, "carbohydrates": 49, "protein": 9, "fiber": 2.7},
            "potato": {"calories": 77, "fat": 0.1, "carbohydrates": 17, "protein": 2, "fiber": 2.2},
            
            # Vegetables
            "tomato": {"calories": 18, "fat": 0.2, "carbohydrates": 3.9, "protein": 0.9, "fiber": 1.2},
            "onion": {"calories": 40, "fat": 0.1, "carbohydrates": 9.3, "protein": 1.1, "fiber": 1.7},
            "carrot": {"calories": 41, "fat": 0.2, "carbohydrates": 9.6, "protein": 0.9, "fiber": 2.8},
            "lettuce": {"calories": 15, "fat": 0.2, "carbohydrates": 2.9, "protein": 1.4, "fiber": 1.3},
            "broccoli": {"calories": 34, "fat": 0.4, "carbohydrates": 7, "protein": 2.8, "fiber": 2.6},
            
            # Dairy
            "cheese": {"calories": 402, "fat": 33, "carbohydrates": 1.3, "protein": 25, "fiber": 0},
            "milk": {"calories": 42, "fat": 1, "carbohydrates": 5, "protein": 3.4, "fiber": 0},
            "butter": {"calories": 717, "fat": 81, "carbohydrates": 0.1, "protein": 0.9, "fiber": 0},
            
            # Fruits
            "apple": {"calories": 52, "fat": 0.2, "carbohydrates": 14, "protein": 0.3, "fiber": 2.4},
            "banana": {"calories": 89, "fat": 0.3, "carbohydrates": 23, "protein": 1.1, "fiber": 2.6},
            
            # Oils and fats
            "oil": {"calories": 884, "fat": 100, "carbohydrates": 0, "protein": 0, "fiber": 0},
        }
    
    def _estimate_portion_size(self, description: str, ingredients: List[str]) -> Tuple[str, float, str]:
        """Estimate portion size based on food description and ingredients."""
        desc_lower = description.lower()
        
        # Portion size patterns
        portion_patterns = {
            'pizza': ('1 slice (medium pizza)', 120, 'slice'),
            'burger': ('1 burger', 200, 'piece'),
            'sandwich': ('1 sandwich', 150, 'piece'),
            'pasta': ('1 cup cooked', 140, 'cup'),
            'rice': ('1 cup cooked', 195, 'cup'),
            'salad': ('1 large bowl', 100, 'bowl'),
            'soup': ('1 cup', 240, 'cup'),
            'steak': ('1 piece (6 oz)', 170, 'piece'),
            'chicken': ('1 breast (4 oz)', 120, 'piece'),
            'fish': ('1 fillet (4 oz)', 120, 'piece'),
            'apple': ('1 medium apple', 180, 'piece'),
            'banana': ('1 medium banana', 120, 'piece'),
        }
        
        # Check description for portion indicators
        for food_type, (portion_desc, weight, serving_type) in portion_patterns.items():
            if food_type in desc_lower:
                return portion_desc, weight, serving_type
        
        # Default portion
        return '1 serving', 150, 'serving'
    
    def _calculate_confidence_range(self, base_confidence: float, portion_certainty: float, ingredient_clarity: float) -> Tuple[str, float, float]:
        """Calculate confidence range based on multiple factors."""
        combined_confidence = (base_confidence + portion_certainty + ingredient_clarity) / 3
        
        if combined_confidence >= 0.9:
            range_percent = 5
        elif combined_confidence >= 0.8:
            range_percent = 8
        elif combined_confidence >= 0.7:
            range_percent = 12
        elif combined_confidence >= 0.6:
            range_percent = 18
        elif combined_confidence >= 0.5:
            range_percent = 25
        elif combined_confidence >= 0.4:
            range_percent = 35
        else:
            range_percent = 50
        
        lower_bound = max(0, combined_confidence - (range_percent / 100))
        upper_bound = min(1.0, combined_confidence + (range_percent / 100))
        range_string = f"±{range_percent}%"
        
        return range_string, lower_bound, upper_bound
    
    def _calculate_per_serving_breakdown(self, nutrition_data: Dict, portion_weight: float) -> Dict[str, Dict[str, float]]:
        """Calculate per-serving nutritional breakdown."""
        base_calories = nutrition_data.get('calories', 0)
        base_fat = nutrition_data.get('fat', 0)
        base_carbs = nutrition_data.get('carbohydrates', 0)
        base_protein = nutrition_data.get('protein', 0)
        
        per_100g_factor = 100 / portion_weight if portion_weight > 0 else 1
        
        return {
            'per_100g': {
                'calories': round(base_calories * per_100g_factor, 1),
                'fat': round(base_fat * per_100g_factor, 1),
                'carbohydrates': round(base_carbs * per_100g_factor, 1),
                'protein': round(base_protein * per_100g_factor, 1)
            },
            'per_serving': {
                'calories': base_calories,
                'fat': base_fat,
                'carbohydrates': base_carbs,
                'protein': base_protein
            },
            'per_half_serving': {
                'calories': round(base_calories * 0.5, 1),
                'fat': round(base_fat * 0.5, 1),
                'carbohydrates': round(base_carbs * 0.5, 1),
                'protein': round(base_protein * 0.5, 1)
            }
        }
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert numerical confidence to confidence level enum."""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_fallback_nutrition(self, ingredients: List[str], description: str) -> Dict:
        """Calculate nutritional values using the local database."""
        total_calories = 0
        total_fat = 0
        total_carbs = 0
        total_protein = 0
        total_fiber = 0
        matched_ingredients = 0
        
        # Look up ingredients in database
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().strip()
            
            for db_food, nutrition in self.nutrition_database.items():
                if db_food in ingredient_lower or ingredient_lower in db_food:
                    portion_factor = 1.0  # 100g
                    
                    total_calories += nutrition['calories'] * portion_factor
                    total_fat += nutrition['fat'] * portion_factor
                    total_carbs += nutrition['carbohydrates'] * portion_factor
                    total_protein += nutrition['protein'] * portion_factor
                    total_fiber += nutrition.get('fiber', 0) * portion_factor
                    
                    matched_ingredients += 1
                    break
        
        # If no ingredients matched, use description-based estimation
        if matched_ingredients == 0:
            desc_lower = description.lower()
            
            if any(word in desc_lower for word in ['pizza', 'pasta', 'bread', 'sandwich']):
                total_calories = 300
                total_fat = 12
                total_carbs = 35
                total_protein = 15
            elif any(word in desc_lower for word in ['salad', 'vegetable', 'green']):
                total_calories = 150
                total_fat = 8
                total_carbs = 15
                total_protein = 8
            elif any(word in desc_lower for word in ['meat', 'chicken', 'beef', 'fish']):
                total_calories = 250
                total_fat = 15
                total_carbs = 5
                total_protein = 25
            else:
                total_calories = 200
                total_fat = 10
                total_carbs = 20
                total_protein = 12
        
        # Calculate confidence based on ingredient matches
        confidence = min(0.8, 0.3 + (matched_ingredients * 0.1))
        
        return {
            'calories': round(total_calories, 1),
            'fat': round(total_fat, 1),
            'carbohydrates': round(total_carbs, 1),
            'protein': round(total_protein, 1),
            'fiber': round(total_fiber, 1),
            'sugar': round(total_carbs * 0.3, 1),
            'sodium': 400,
            'confidence': confidence,
            'reasoning': f'Database lookup with {matched_ingredients} matched ingredients'
        }
    
    def calculate_nutrition(self, ingredients: List[str], description: str, portion_info: str = "") -> NutritionalValues:
        """Calculate nutritional values from ingredients and description."""
        try:
            print(f"Calculating nutrition for: {description}")
            print(f"Ingredients: {ingredients}")
            
            # Estimate portion size
            portion_desc, portion_weight, serving_type = self._estimate_portion_size(description, ingredients)
            if not portion_info:
                portion_info = portion_desc
            
            # Assess ingredient clarity
            ingredient_clarity = min(1.0, len(ingredients) * 0.2) if ingredients else 0.3
            
            # Calculate nutrition using fallback method
            nutrition_data = self._calculate_fallback_nutrition(ingredients, description)
            
            # Add portion information
            nutrition_data['portion_size'] = portion_desc
            nutrition_data['portion_weight_grams'] = portion_weight
            nutrition_data['portion_certainty'] = 0.7
            
            # Calculate confidence metrics
            base_confidence = nutrition_data.get('confidence', 0.5)
            portion_certainty = nutrition_data.get('portion_certainty', 0.6)
            
            confidence_range, lower_bound, upper_bound = self._calculate_confidence_range(
                base_confidence, portion_certainty, ingredient_clarity
            )
            
            # Calculate per-serving breakdown
            per_serving_breakdown = self._calculate_per_serving_breakdown(nutrition_data, portion_weight)
            
            # Create result object
            confidence_level = self._determine_confidence_level(base_confidence)
            
            nutritional_values = NutritionalValues(
                calories=nutrition_data.get('calories', 200),
                fat=nutrition_data.get('fat', 10),
                carbohydrates=nutrition_data.get('carbohydrates', 20),
                protein=nutrition_data.get('protein', 10),
                fiber=nutrition_data.get('fiber', 2),
                sugar=nutrition_data.get('sugar', 5),
                sodium=nutrition_data.get('sodium', 300),
                confidence=base_confidence,
                confidence_level=confidence_level,
                portion_size=portion_desc,
                portion_weight_grams=portion_weight,
                portion_certainty=portion_certainty,
                confidence_range=confidence_range,
                confidence_lower_bound=lower_bound,
                confidence_upper_bound=upper_bound,
                per_serving_breakdown=per_serving_breakdown
            )
            
            print(f"Nutrition calculation completed with {confidence_level.value} confidence")
            return nutritional_values
            
        except Exception as e:
            print(f"Nutrition calculation failed: {e}")
            
            # Return safe defaults
            default_portion_desc, default_portion_weight, _ = self._estimate_portion_size(description, ingredients)
            
            return NutritionalValues(
                calories=200,
                fat=10,
                carbohydrates=20,
                protein=10,
                fiber=2,
                sugar=5,
                sodium=300,
                confidence=0.2,
                confidence_level=ConfidenceLevel.VERY_LOW,
                portion_size=default_portion_desc,
                portion_weight_grams=default_portion_weight,
                portion_certainty=0.3,
                confidence_range="±50%",
                confidence_lower_bound=0.0,
                confidence_upper_bound=0.4,
                per_serving_breakdown={}
            )
    
    def analyze_nutrition(self, vision_results: Dict) -> Dict:
        """Analyze nutrition from vision model results."""
        try:
            ingredients = vision_results.get('ingredients', [])
            description = vision_results.get('description', 'Unknown food item')
            
            nutrition = self.calculate_nutrition(ingredients, description)
            
            return {
                'calories': nutrition.calories,
                'fat': nutrition.fat,
                'carbohydrates': nutrition.carbohydrates,
                'protein': nutrition.protein,
                'fiber': nutrition.fiber,
                'sugar': nutrition.sugar,
                'sodium': nutrition.sodium,
                'portion_size': nutrition.portion_size,
                'portion_weight_grams': nutrition.portion_weight_grams,
                'portion_certainty': nutrition.portion_certainty,
                'confidence': nutrition.confidence,
                'confidence_level': nutrition.confidence_level.value,
                'confidence_range': nutrition.confidence_range,
                'confidence_bounds': {
                    'lower': nutrition.confidence_lower_bound,
                    'upper': nutrition.confidence_upper_bound
                },
                'per_serving_breakdown': nutrition.per_serving_breakdown,
                'analysis_status': 'success',
                'timestamp': None
            }
            
        except Exception as e:
            print(f"Nutrition analysis failed: {e}")
            return {
                'calories': 0,
                'fat': 0,
                'carbohydrates': 0,
                'protein': 0,
                'fiber': 0,
                'sugar': 0,
                'sodium': 0,
                'portion_size': '1 serving',
                'portion_weight_grams': 150,
                'portion_certainty': 0.2,
                'confidence': 0.0,
                'confidence_level': 'very_low',
                'confidence_range': '±50%',
                'confidence_bounds': {'lower': 0.0, 'upper': 0.4},
                'per_serving_breakdown': {},
                'analysis_status': 'error',
                'error_message': str(e)
            }


# Convenience function
def analyze_nutrition_from_vision(vision_results: Dict, model_name: str = "fallback") -> Dict:
    """Quick utility function to analyze nutrition from vision results."""
    nutrition_llm = NutritionLLM(model_name=model_name)
    return nutrition_llm.analyze_nutrition(vision_results)