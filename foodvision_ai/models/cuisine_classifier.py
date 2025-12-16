"""
Cuisine Classifier for FoodVisionAI

Implements cuisine classification using keyword-based and BERT-based approaches
to identify cuisine types and cultural context from food ingredients and descriptions.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import json
import re
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import math

# from ..config import settings  # Commented out to avoid import issues during testing


class CuisineConfidenceLevel(Enum):
    """Confidence levels for cuisine classification."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class CuisineResult:
    """Single cuisine classification result with confidence."""
    name: str
    confidence: float
    confidence_level: CuisineConfidenceLevel
    reasoning: str
    cultural_context: Optional[str] = None


@dataclass
class CuisineClassification:
    """Complete cuisine classification results."""
    primary_cuisine: CuisineResult
    all_cuisines: List[CuisineResult]
    multiple_cuisines_detected: bool
    fusion_detected: bool
    uncertainty_level: str
    total_confidence: float
    cultural_analysis: Dict[str, Any]


class CuisineClassifier:
    """
    Cuisine classifier for identifying cuisine types and cultural context.
    
    Uses a combination of:
    - Keyword-based classification for fast, reliable detection
    - BERT-based semantic analysis for complex cases
    - Cultural context database for enhanced analysis
    """
    
    def __init__(self, use_bert: bool = True, device: Optional[str] = None):
        """
        Initialize the Cuisine Classifier.
        
        Args:
            use_bert: Whether to use BERT for semantic analysis
            device: Device to run models on ('cpu', 'cuda', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_bert = use_bert
        logger.info(f"Initializing Cuisine Classifier on device: {self.device}")
        
        # Initialize BERT model if requested
        self.bert_tokenizer = None
        self.bert_model = None
        
        # Load cuisine databases
        self.cuisine_keywords = self._load_cuisine_keywords()
        self.ingredient_cuisine_map = self._load_ingredient_cuisine_mapping()
        self.cultural_context_db = self._load_cultural_context_database()
        self.fusion_patterns = self._load_fusion_patterns()
        
        # Initialize BERT model if requested
        if self.use_bert:
            self._load_bert_model()
    
    def _load_bert_model(self):
        """Load BERT model for semantic cuisine analysis."""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Loading BERT model: {model_name}")
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            logger.success("BERT model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            logger.info("Falling back to keyword-only classification")
            self.use_bert = False
    
    def _load_cuisine_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load cuisine-specific keywords and indicators.
        
        Returns:
            Dictionary mapping cuisines to their characteristic keywords
        """
        cuisine_keywords = {
            "Italian": {
                "ingredients": [
                    "pasta", "spaghetti", "linguine", "penne", "ravioli", "gnocchi",
                    "parmesan", "mozzarella", "ricotta", "prosciutto", "pancetta",
                    "basil", "oregano", "tomato", "olive oil", "balsamic",
                    "risotto", "arborio rice", "pine nuts", "garlic", "rosemary"
                ],
                "dishes": [
                    "pizza", "lasagna", "carbonara", "bolognese", "marinara",
                    "pesto", "tiramisu", "gelato", "bruschetta", "caprese",
                    "osso buco", "saltimbocca", "parmigiana", "focaccia"
                ],
                "techniques": ["al dente", "soffritto", "risotto", "antipasto"]
            },
            "Chinese": {
                "ingredients": [
                    "soy sauce", "rice", "noodles", "bok choy", "shiitake",
                    "ginger", "garlic", "scallion", "sesame oil", "rice vinegar",
                    "star anise", "five spice", "hoisin", "oyster sauce",
                    "tofu", "bean sprouts", "water chestnuts", "bamboo shoots"
                ],
                "dishes": [
                    "fried rice", "chow mein", "dim sum", "dumplings", "spring rolls",
                    "kung pao", "sweet and sour", "mapo tofu", "peking duck",
                    "hot pot", "wontons", "bao", "congee"
                ],
                "techniques": ["stir fry", "steaming", "braising", "wok"]
            },
            "Mexican": {
                "ingredients": [
                    "corn", "beans", "rice", "avocado", "lime", "cilantro",
                    "jalapeño", "chipotle", "cumin", "chili powder", "paprika",
                    "tomato", "onion", "garlic", "cheese", "sour cream",
                    "masa", "tortilla", "salsa", "guacamole", "queso"
                ],
                "dishes": [
                    "tacos", "burritos", "quesadillas", "enchiladas", "tamales",
                    "nachos", "fajitas", "carnitas", "barbacoa", "pozole",
                    "mole", "churros", "tres leches", "elote"
                ],
                "techniques": ["grilling", "roasting", "braising", "charring"]
            },
            "Indian": {
                "ingredients": [
                    "curry", "turmeric", "cumin", "coriander", "cardamom",
                    "cinnamon", "cloves", "garam masala", "ginger", "garlic",
                    "onion", "tomato", "coconut", "yogurt", "ghee",
                    "basmati rice", "lentils", "chickpeas", "paneer", "naan"
                ],
                "dishes": [
                    "curry", "biryani", "tandoori", "masala", "dal", "samosa",
                    "pakora", "chutney", "raita", "korma", "vindaloo",
                    "butter chicken", "palak paneer", "rogan josh"
                ],
                "techniques": ["tandooring", "tempering", "slow cooking", "grinding spices"]
            },
            "Japanese": {
                "ingredients": [
                    "rice", "soy sauce", "miso", "sake", "mirin", "dashi",
                    "nori", "wasabi", "ginger", "sesame", "tofu", "shiitake",
                    "edamame", "cucumber", "avocado", "tuna", "salmon"
                ],
                "dishes": [
                    "sushi", "sashimi", "ramen", "udon", "soba", "tempura",
                    "teriyaki", "yakitori", "miso soup", "onigiri", "katsu",
                    "donburi", "takoyaki", "okonomiyaki", "mochi"
                ],
                "techniques": ["sushi making", "tempura frying", "grilling", "steaming"]
            },
            "French": {
                "ingredients": [
                    "butter", "cream", "wine", "herbs", "shallots", "garlic",
                    "mushrooms", "cheese", "brie", "camembert", "gruyere",
                    "thyme", "rosemary", "tarragon", "chives", "parsley"
                ],
                "dishes": [
                    "coq au vin", "bouillabaisse", "ratatouille", "quiche",
                    "croissant", "baguette", "soufflé", "crème brûlée",
                    "escargot", "foie gras", "cassoulet", "confit"
                ],
                "techniques": ["sautéing", "braising", "flambéing", "reduction"]
            },
            "Thai": {
                "ingredients": [
                    "coconut milk", "lemongrass", "galangal", "kaffir lime",
                    "fish sauce", "palm sugar", "thai basil", "chili",
                    "garlic", "shallots", "lime", "peanuts", "rice noodles"
                ],
                "dishes": [
                    "pad thai", "green curry", "red curry", "tom yum", "som tam",
                    "massaman", "pad krapow", "mango sticky rice", "larb"
                ],
                "techniques": ["stir frying", "curry making", "balancing flavors"]
            },
            "Mediterranean": {
                "ingredients": [
                    "olive oil", "lemon", "garlic", "herbs", "tomato", "cucumber",
                    "olives", "feta", "yogurt", "chickpeas", "tahini", "za'atar"
                ],
                "dishes": [
                    "hummus", "tabbouleh", "falafel", "greek salad", "moussaka",
                    "baklava", "dolmas", "tzatziki", "spanakopita"
                ],
                "techniques": ["grilling", "roasting", "marinating"]
            },
            "Korean": {
                "ingredients": [
                    "kimchi", "gochujang", "sesame oil", "soy sauce", "garlic",
                    "ginger", "scallions", "rice", "noodles", "tofu", "beef"
                ],
                "dishes": [
                    "bibimbap", "bulgogi", "kimchi", "japchae", "galbi",
                    "tteokbokki", "banchan", "korean bbq"
                ],
                "techniques": ["fermentation", "grilling", "stir frying"]
            },
            "Middle Eastern": {
                "ingredients": [
                    "tahini", "chickpeas", "lentils", "bulgur", "lamb", "yogurt",
                    "mint", "parsley", "sumac", "za'atar", "pomegranate", "dates"
                ],
                "dishes": [
                    "hummus", "falafel", "shawarma", "kebab", "tabbouleh",
                    "baba ganoush", "pilaf", "baklava", "halva"
                ],
                "techniques": ["grilling", "roasting", "slow cooking"]
            }
        }
        
        return cuisine_keywords
    
    def _load_ingredient_cuisine_mapping(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Load ingredient to cuisine mapping with confidence weights.
        
        Returns:
            Dictionary mapping ingredients to list of (cuisine, weight) tuples
        """
        ingredient_map = {}
        
        # Build reverse mapping from cuisine keywords
        for cuisine, categories in self.cuisine_keywords.items():
            for category, items in categories.items():
                weight = 1.0 if category == "ingredients" else 0.8 if category == "dishes" else 0.6
                
                for item in items:
                    if item not in ingredient_map:
                        ingredient_map[item] = []
                    ingredient_map[item].append((cuisine, weight))
        
        # Add specific high-confidence mappings
        high_confidence_mappings = {
            "pasta": [("Italian", 0.95)],
            "soy sauce": [("Chinese", 0.9), ("Japanese", 0.8)],
            "curry": [("Indian", 0.9), ("Thai", 0.7)],
            "kimchi": [("Korean", 0.98)],
            "miso": [("Japanese", 0.95)],
            "garam masala": [("Indian", 0.98)],
            "fish sauce": [("Thai", 0.9), ("Vietnamese", 0.8)],
            "tahini": [("Middle Eastern", 0.9), ("Mediterranean", 0.7)],
            "tortilla": [("Mexican", 0.95)],
            "baguette": [("French", 0.9)],
            "olive oil": [("Mediterranean", 0.8), ("Italian", 0.7)],
            "coconut milk": [("Thai", 0.8), ("Indian", 0.6)],
            "lemongrass": [("Thai", 0.95), ("Vietnamese", 0.8)],
            "wasabi": [("Japanese", 0.98)],
            "gochujang": [("Korean", 0.98)],
            "harissa": [("North African", 0.95), ("Middle Eastern", 0.7)]
        }
        
        # Update with high-confidence mappings
        for ingredient, mappings in high_confidence_mappings.items():
            ingredient_map[ingredient] = mappings
        
        return ingredient_map
    
    def _load_cultural_context_database(self) -> Dict[str, Dict[str, Any]]:
        """
        Load cultural context information for cuisines.
        
        Returns:
            Dictionary with cultural context for each cuisine
        """
        cultural_context = {
            "Italian": {
                "regions": ["Northern Italy", "Southern Italy", "Sicily", "Tuscany", "Emilia-Romagna"],
                "characteristics": ["Fresh ingredients", "Regional diversity", "Pasta and rice", "Wine culture"],
                "cooking_methods": ["Slow cooking", "Fresh herbs", "Olive oil based", "Seasonal ingredients"],
                "meal_structure": ["Antipasto", "Primo", "Secondo", "Dolce"],
                "cultural_notes": "Emphasis on family meals and regional traditions"
            },
            "Chinese": {
                "regions": ["Sichuan", "Cantonese", "Hunan", "Jiangsu", "Shandong", "Fujian", "Anhui", "Zhejiang"],
                "characteristics": ["Balance of flavors", "Texture variety", "Seasonal eating", "Medicinal properties"],
                "cooking_methods": ["Stir-frying", "Steaming", "Braising", "Deep-frying"],
                "meal_structure": ["Shared dishes", "Rice or noodles", "Tea culture"],
                "cultural_notes": "Philosophy of food as medicine and harmony"
            },
            "Mexican": {
                "regions": ["Oaxaca", "Yucatan", "Puebla", "Jalisco", "Veracruz"],
                "characteristics": ["Corn-based", "Spicy flavors", "Indigenous ingredients", "Festive foods"],
                "cooking_methods": ["Grilling", "Roasting", "Slow cooking", "Fermentation"],
                "meal_structure": ["Breakfast", "Comida", "Cena", "Street food"],
                "cultural_notes": "Rich indigenous heritage mixed with Spanish influences"
            },
            "Indian": {
                "regions": ["North Indian", "South Indian", "Bengali", "Gujarati", "Punjabi", "Rajasthani"],
                "characteristics": ["Spice complexity", "Vegetarian options", "Regional diversity", "Ayurvedic principles"],
                "cooking_methods": ["Tempering spices", "Slow cooking", "Tandoor", "Steaming"],
                "meal_structure": ["Thali system", "Multiple courses", "Sweet endings"],
                "cultural_notes": "Food as spiritual and medicinal practice"
            },
            "Japanese": {
                "regions": ["Kanto", "Kansai", "Kyushu", "Tohoku", "Hokkaido"],
                "characteristics": ["Seasonal ingredients", "Minimal processing", "Aesthetic presentation", "Umami focus"],
                "cooking_methods": ["Grilling", "Steaming", "Raw preparation", "Fermentation"],
                "meal_structure": ["Ichijuu sansai", "Seasonal kaiseki", "Bento culture"],
                "cultural_notes": "Harmony with nature and seasonal awareness"
            },
            "French": {
                "regions": ["Provence", "Burgundy", "Normandy", "Alsace", "Loire Valley"],
                "characteristics": ["Technique mastery", "Sauce expertise", "Wine pairing", "Seasonal cooking"],
                "cooking_methods": ["Classical techniques", "Sauce making", "Pastry arts", "Charcuterie"],
                "meal_structure": ["Multiple courses", "Wine service", "Cheese course"],
                "cultural_notes": "Culinary arts as high culture and gastronomy"
            },
            "Thai": {
                "regions": ["Central Thailand", "Northern Thailand", "Northeastern Thailand", "Southern Thailand"],
                "characteristics": ["Balance of sweet, sour, salty, spicy", "Fresh herbs", "Coconut milk", "Rice culture"],
                "cooking_methods": ["Stir-frying", "Curry making", "Grilling", "Steaming"],
                "meal_structure": ["Shared dishes", "Rice centerpiece", "Fresh fruits"],
                "cultural_notes": "Buddhist influences and tropical abundance"
            },
            "Mediterranean": {
                "regions": ["Greek", "Turkish", "Lebanese", "Moroccan", "Spanish"],
                "characteristics": ["Olive oil", "Fresh vegetables", "Seafood", "Healthy fats"],
                "cooking_methods": ["Grilling", "Roasting", "Marinating", "Simple preparation"],
                "meal_structure": ["Mezze culture", "Shared plates", "Seasonal eating"],
                "cultural_notes": "Ancient traditions and healthy lifestyle"
            },
            "Korean": {
                "regions": ["Seoul", "Jeolla", "Gyeongsang", "Gangwon", "Jeju"],
                "characteristics": ["Fermented foods", "Spicy flavors", "Banchan culture", "Seasonal kimchi"],
                "cooking_methods": ["Fermentation", "Grilling", "Stewing", "Pickling"],
                "meal_structure": ["Rice and soup", "Multiple banchan", "Communal eating"],
                "cultural_notes": "Fermentation mastery and communal dining"
            },
            "Middle Eastern": {
                "regions": ["Levantine", "Persian", "Turkish", "Arabian", "North African"],
                "characteristics": ["Spice blends", "Grains and legumes", "Hospitality foods", "Ancient grains"],
                "cooking_methods": ["Slow cooking", "Grilling", "Stuffing", "Spice roasting"],
                "meal_structure": ["Mezze", "Main dishes", "Sweet endings", "Tea culture"],
                "cultural_notes": "Hospitality traditions and spice trade heritage"
            }
        }
        
        return cultural_context
    
    def _load_fusion_patterns(self) -> Dict[str, List[str]]:
        """
        Load patterns that indicate fusion cuisine.
        
        Returns:
            Dictionary of fusion patterns and their indicators
        """
        fusion_patterns = {
            "Asian_Fusion": [
                "soy sauce + cheese", "kimchi + burger", "ramen + western",
                "sushi + non-japanese", "curry + pasta", "miso + western"
            ],
            "Tex_Mex": [
                "mexican + american", "cheese + mexican", "beef + mexican spices",
                "nachos + non-mexican", "burrito + american"
            ],
            "Mediterranean_Fusion": [
                "hummus + non-middle-eastern", "feta + non-greek",
                "olive oil + asian", "tahini + western"
            ],
            "Italian_American": [
                "pasta + american cheese", "pizza + american toppings",
                "italian + heavy cream", "garlic bread + american"
            ],
            "French_Asian": [
                "french technique + asian ingredients", "soy + french sauce",
                "asian + butter", "french + ginger"
            ],
            "Modern_Fusion": [
                "molecular + traditional", "deconstructed + classic",
                "foam + ethnic", "spherification + traditional"
            ]
        }
        
        return fusion_patterns
    
    def _calculate_keyword_scores(self, ingredients: List[str], description: str) -> Dict[str, float]:
        """
        Calculate cuisine scores based on keyword matching.
        
        Args:
            ingredients: List of ingredients
            description: Food description
            
        Returns:
            Dictionary of cuisine scores
        """
        cuisine_scores = defaultdict(float)
        text_to_analyze = " ".join(ingredients + [description]).lower()
        
        # Score based on ingredient mapping
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().strip()
            
            # Direct ingredient matches
            if ingredient_lower in self.ingredient_cuisine_map:
                for cuisine, weight in self.ingredient_cuisine_map[ingredient_lower]:
                    cuisine_scores[cuisine] += weight
            
            # Partial matches for compound ingredients
            for mapped_ingredient, cuisine_weights in self.ingredient_cuisine_map.items():
                if mapped_ingredient in ingredient_lower or ingredient_lower in mapped_ingredient:
                    for cuisine, weight in cuisine_weights:
                        cuisine_scores[cuisine] += weight * 0.7  # Reduced weight for partial matches
        
        # Score based on keyword presence in description
        for cuisine, categories in self.cuisine_keywords.items():
            for category, keywords in categories.items():
                category_weight = 1.0 if category == "ingredients" else 0.8 if category == "dishes" else 0.6
                
                for keyword in keywords:
                    if keyword.lower() in text_to_analyze:
                        cuisine_scores[cuisine] += category_weight
                        
                        # Bonus for exact matches
                        if keyword.lower() in [ing.lower() for ing in ingredients]:
                            cuisine_scores[cuisine] += 0.3
        
        # Normalize scores
        if cuisine_scores:
            max_score = max(cuisine_scores.values())
            if max_score > 0:
                for cuisine in cuisine_scores:
                    cuisine_scores[cuisine] = cuisine_scores[cuisine] / max_score
        
        return dict(cuisine_scores)
    
    def _detect_fusion_cuisine(self, ingredients: List[str], description: str, cuisine_scores: Dict[str, float]) -> Tuple[bool, List[str], str]:
        """
        Detect if the food represents fusion cuisine.
        
        Args:
            ingredients: List of ingredients
            description: Food description
            cuisine_scores: Calculated cuisine scores
            
        Returns:
            Tuple of (is_fusion, fusion_types, fusion_description)
        """
        text_to_analyze = " ".join(ingredients + [description]).lower()
        
        # Check for multiple high-scoring cuisines (indicator of fusion)
        high_scoring_cuisines = [cuisine for cuisine, score in cuisine_scores.items() if score > 0.4]
        
        fusion_detected = len(high_scoring_cuisines) > 1
        fusion_types = []
        fusion_description = ""
        
        if fusion_detected:
            # Check for specific fusion patterns
            detected_patterns = []
            
            for fusion_type, patterns in self.fusion_patterns.items():
                for pattern in patterns:
                    pattern_parts = pattern.split(" + ")
                    if all(part.lower() in text_to_analyze for part in pattern_parts):
                        detected_patterns.append(fusion_type)
                        break
            
            if detected_patterns:
                fusion_types = detected_patterns
                fusion_description = f"Fusion cuisine combining {' and '.join(high_scoring_cuisines)}"
            else:
                fusion_types = [f"{high_scoring_cuisines[0]}_Fusion"]
                fusion_description = f"Fusion of {' and '.join(high_scoring_cuisines)} elements"
        
        return fusion_detected, fusion_types, fusion_description
    
    def _calculate_bert_similarity(self, text: str, cuisine_descriptions: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate semantic similarity using BERT embeddings.
        
        Args:
            text: Text to analyze (ingredients + description)
            cuisine_descriptions: Descriptions of each cuisine for comparison
            
        Returns:
            Dictionary of similarity scores for each cuisine
        """
        if not self.use_bert or not self.bert_model:
            return {}
        
        try:
            # Tokenize input text
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings for input text
            with torch.no_grad():
                text_outputs = self.bert_model(**inputs)
                text_embedding = text_outputs.last_hidden_state.mean(dim=1)
            
            similarity_scores = {}
            
            # Calculate similarity with each cuisine description
            for cuisine, description in cuisine_descriptions.items():
                cuisine_inputs = self.bert_tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=512)
                cuisine_inputs = {k: v.to(self.device) for k, v in cuisine_inputs.items()}
                
                with torch.no_grad():
                    cuisine_outputs = self.bert_model(**cuisine_inputs)
                    cuisine_embedding = cuisine_outputs.last_hidden_state.mean(dim=1)
                
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(text_embedding, cuisine_embedding)
                similarity_scores[cuisine] = similarity.item()
            
            return similarity_scores
            
        except Exception as e:
            logger.warning(f"BERT similarity calculation failed: {e}")
            return {}
    
    def _determine_confidence_level(self, confidence_score: float) -> CuisineConfidenceLevel:
        """
        Convert numerical confidence to confidence level enum.
        
        Args:
            confidence_score: Numerical confidence (0.0-1.0)
            
        Returns:
            CuisineConfidenceLevel enum value
        """
        if confidence_score >= 0.9:
            return CuisineConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return CuisineConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return CuisineConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return CuisineConfidenceLevel.LOW
        else:
            return CuisineConfidenceLevel.VERY_LOW
    
    def _generate_reasoning(self, cuisine: str, ingredients: List[str], description: str, score: float) -> str:
        """
        Generate reasoning for cuisine classification.
        
        Args:
            cuisine: Classified cuisine name
            ingredients: List of ingredients
            description: Food description
            score: Confidence score
            
        Returns:
            Human-readable reasoning string
        """
        matched_ingredients = []
        matched_dishes = []
        
        # Find matching elements
        text_lower = " ".join(ingredients + [description]).lower()
        
        if cuisine in self.cuisine_keywords:
            cuisine_data = self.cuisine_keywords[cuisine]
            
            # Check ingredients
            for ingredient in cuisine_data.get("ingredients", []):
                if ingredient.lower() in text_lower:
                    matched_ingredients.append(ingredient)
            
            # Check dishes
            for dish in cuisine_data.get("dishes", []):
                if dish.lower() in text_lower:
                    matched_dishes.append(dish)
        
        # Build reasoning
        reasoning_parts = []
        
        if matched_ingredients:
            reasoning_parts.append(f"Contains {cuisine} ingredients: {', '.join(matched_ingredients[:3])}")
        
        if matched_dishes:
            reasoning_parts.append(f"Matches {cuisine} dishes: {', '.join(matched_dishes[:2])}")
        
        if score > 0.8:
            reasoning_parts.append("Strong cultural indicators present")
        elif score > 0.6:
            reasoning_parts.append("Multiple cuisine markers identified")
        elif score > 0.4:
            reasoning_parts.append("Some characteristic elements found")
        else:
            reasoning_parts.append("Limited cuisine indicators")
        
        return ". ".join(reasoning_parts) if reasoning_parts else f"Classified as {cuisine} based on available evidence"
    
    def _handle_uncertainty(self, cuisine_scores: Dict[str, float], ingredients: List[str], description: str) -> Tuple[str, float]:
        """
        Handle cases where cuisine cannot be determined with confidence.
        
        Args:
            cuisine_scores: Calculated cuisine scores
            ingredients: List of ingredients
            description: Food description
            
        Returns:
            Tuple of (uncertainty_level, adjusted_confidence)
        """
        if not cuisine_scores:
            return "very_high", 0.1
        
        max_score = max(cuisine_scores.values())
        score_spread = max_score - min(cuisine_scores.values()) if len(cuisine_scores) > 1 else max_score
        num_ingredients = len(ingredients)
        description_quality = len(description.split()) if description else 0
        
        # Calculate uncertainty factors
        ingredient_factor = min(1.0, num_ingredients / 5.0)  # More ingredients = less uncertainty
        description_factor = min(1.0, description_quality / 10.0)  # Better description = less uncertainty
        score_confidence_factor = max_score
        spread_factor = score_spread  # Higher spread = more confidence in top choicetor = score_spread if len(cuisine_scores) > 1 else 1.0
        
        # Combined uncertainty assessment
        combined_confidence = (ingredient_factor + description_factor + score_confidence_factor + spread_factor) / 4
        
        # Determine uncertainty level with enhanced logic
        if max_score < 0.2 or combined_confidence < 0.3:
            uncertainty_level = "very_high"
            adjusted_confidence = max_score * 0.4
        elif max_score < 0.4 or combined_confidence < 0.5:
            uncertainty_level = "high"
            adjusted_confidence = max_score * 0.6
        elif score_spread < 0.15 or combined_confidence < 0.7:
            uncertainty_level = "medium"
            adjusted_confidence = max_score * 0.8
        elif max_score < 0.7 or combined_confidence < 0.85:
            uncertainty_level = "low"
            adjusted_confidence = max_score * 0.9
        else:
            uncertainty_level = "very_low"
            adjusted_confidence = min(max_score * 1.1, 1.0)
        
        return uncertainty_level, adjusted_confidence
    
    def _calculate_uncertainty_indicators(self, ingredients: List[str], description: str, cuisine_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate comprehensive uncertainty indicators for cuisine classification.
        
        Args:
            ingredients: List of ingredients
            description: Food description
            cuisine_scores: Calculated cuisine scores
            
        Returns:
            Dictionary with detailed uncertainty analysis
        """
        # Basic metrics
        num_ingredients = len(ingredients)
        description_length = len(description.split()) if description else 0
        
        # Score analysis
        if not cuisine_scores:
            max_score = 0.0
            score_variance = 0.0
            top_scores_ratio = 0.0
        else:
            scores = list(cuisine_scores.values())
            max_score = max(scores)
            score_variance = np.var(scores) if len(scores) > 1 else 0.0
            
            # Ratio of top 2 scores (lower ratio = more uncertainty)
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
                top_scores_ratio = sorted_scores[1] / sorted_scores[0]
            else:
                top_scores_ratio = 0.0
        
        # Ingredient specificity analysis
        ambiguity_assessment = self._assess_ambiguous_ingredients(ingredients)
        
        # Calculate uncertainty factors
        uncertainty_factors = {
            "ingredient_count": {
                "value": num_ingredients,
                "uncertainty": max(0.0, 1.0 - (num_ingredients / 8.0)),  # Optimal around 8 ingredients
                "weight": 0.2
            },
            "description_quality": {
                "value": description_length,
                "uncertainty": max(0.0, 1.0 - (description_length / 15.0)),  # Optimal around 15 words
                "weight": 0.15
            },
            "score_confidence": {
                "value": max_score,
                "uncertainty": 1.0 - max_score,
                "weight": 0.3
            },
            "score_separation": {
                "value": 1.0 - top_scores_ratio,
                "uncertainty": top_scores_ratio,  # High ratio = high uncertainty
                "weight": 0.2
            },
            "ingredient_ambiguity": {
                "value": ambiguity_assessment["clarity_score"],
                "uncertainty": ambiguity_assessment["ambiguity_score"],
                "weight": 0.15
            }
        }
        
        # Calculate weighted uncertainty
        total_uncertainty = 0.0
        total_weight = 0.0
        
        for factor_name, factor_data in uncertainty_factors.items():
            uncertainty_contribution = factor_data["uncertainty"] * factor_data["weight"]
            total_uncertainty += uncertainty_contribution
            total_weight += factor_data["weight"]
        
        overall_uncertainty = total_uncertainty / total_weight if total_weight > 0 else 1.0
        
        # Determine uncertainty category
        if overall_uncertainty >= 0.8:
            uncertainty_category = "very_high"
            confidence_multiplier = 0.2
        elif overall_uncertainty >= 0.6:
            uncertainty_category = "high"
            confidence_multiplier = 0.4
        elif overall_uncertainty >= 0.4:
            uncertainty_category = "medium"
            confidence_multiplier = 0.6
        elif overall_uncertainty >= 0.2:
            uncertainty_category = "low"
            confidence_multiplier = 0.8
        else:
            uncertainty_category = "very_low"
            confidence_multiplier = 1.0
        
        return {
            "overall_uncertainty": overall_uncertainty,
            "uncertainty_category": uncertainty_category,
            "confidence_multiplier": confidence_multiplier,
            "uncertainty_factors": uncertainty_factors,
            "recommendations": self._generate_uncertainty_recommendations(uncertainty_factors),
            "should_request_more_info": overall_uncertainty > 0.7
        }
    
    def _generate_uncertainty_recommendations(self, uncertainty_factors: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for reducing uncertainty.
        
        Args:
            uncertainty_factors: Dictionary of uncertainty factors
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check each factor and provide specific recommendations
        if uncertainty_factors["ingredient_count"]["uncertainty"] > 0.5:
            recommendations.append("More ingredient details would improve classification accuracy")
        
        if uncertainty_factors["description_quality"]["uncertainty"] > 0.5:
            recommendations.append("A more detailed food description would help identify cuisine type")
        
        if uncertainty_factors["score_confidence"]["uncertainty"] > 0.6:
            recommendations.append("No strong cuisine indicators found - consider providing cooking method or origin")
        
        if uncertainty_factors["score_separation"]["uncertainty"] > 0.7:
            recommendations.append("Multiple cuisine influences detected - this may be fusion cuisine")
        
        if uncertainty_factors["ingredient_ambiguity"]["uncertainty"] > 0.6:
            recommendations.append("Ingredients are common across multiple cuisines - specific spices or techniques would help")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Classification confidence is good based on available information")
        
        return recommendations
    
    def _apply_confidence_weighting(self, base_confidence: float, uncertainty_indicators: Dict[str, Any]) -> float:
        """
        Apply sophisticated confidence weighting based on uncertainty analysis.
        
        Args:
            base_confidence: Base confidence score
            uncertainty_indicators: Uncertainty analysis results
            
        Returns:
            Adjusted confidence score
        """
        confidence_multiplier = uncertainty_indicators["confidence_multiplier"]
        
        # Apply base multiplier
        adjusted_confidence = base_confidence * confidence_multiplier
        
        # Additional adjustments based on specific factors
        factors = uncertainty_indicators["uncertainty_factors"]
        
        # Bonus for high ingredient count and quality description
        if factors["ingredient_count"]["uncertainty"] < 0.3 and factors["description_quality"]["uncertainty"] < 0.3:
            adjusted_confidence *= 1.1  # 10% bonus for good input quality
        
        # Penalty for very ambiguous ingredients
        if factors["ingredient_ambiguity"]["uncertainty"] > 0.8:
            adjusted_confidence *= 0.8  # 20% penalty for very ambiguous ingredients
        
        # Bonus for clear score separation
        if factors["score_separation"]["uncertainty"] < 0.2:
            adjusted_confidence *= 1.05  # 5% bonus for clear winner
        
        # Ensure confidence stays within bounds
        return min(max(adjusted_confidence, 0.0), 1.0)
    
    def _create_fallback_classification(self, ingredients: List[str], description: str, error_msg: str = None) -> CuisineResult:
        """
        Create fallback classification when cuisine cannot be determined.
        
        Args:
            ingredients: List of ingredients
            description: Food description
            error_msg: Optional error message
            
        Returns:
            CuisineResult with fallback classification
        """
        # Try to make educated guesses based on common patterns
        text_lower = " ".join(ingredients + [description]).lower()
        
        # Basic pattern matching for fallback
        fallback_patterns = {
            "Western": ["cheese", "butter", "cream", "beef", "chicken", "potato", "bread"],
            "Asian": ["rice", "soy", "ginger", "garlic", "noodles", "sesame"],
            "Mediterranean": ["olive", "tomato", "herb", "lemon", "fish"],
            "Vegetarian": ["vegetable", "bean", "lentil", "quinoa", "tofu"],
            "Comfort Food": ["fried", "creamy", "hearty", "rich", "indulgent"]
        }
        
        fallback_scores = {}
        for category, keywords in fallback_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                fallback_scores[category] = score / len(keywords)
        
        if fallback_scores:
            best_fallback = max(fallback_scores.items(), key=lambda x: x[1])
            confidence = min(0.3, best_fallback[1])  # Cap fallback confidence at 0.3
            reasoning = f"Fallback classification based on general food patterns. {error_msg or ''}"
        else:
            best_fallback = ("Unknown", 0.1)
            confidence = 0.1
            reasoning = f"Unable to classify cuisine type. {error_msg or 'Insufficient distinctive ingredients or description.'}"
        
        return CuisineResult(
            name=best_fallback[0],
            confidence=confidence,
            confidence_level=self._determine_confidence_level(confidence),
            reasoning=reasoning.strip(),
            cultural_context="General or mixed cuisine elements"
        )
    
    def _assess_ambiguous_ingredients(self, ingredients: List[str]) -> Dict[str, Any]:
        """
        Assess ingredients for ambiguity and provide uncertainty indicators.
        
        Args:
            ingredients: List of ingredients
            
        Returns:
            Dictionary with ambiguity assessment
        """
        ambiguous_ingredients = []
        clear_indicators = []
        
        # Common ambiguous ingredients (used in multiple cuisines)
        ambiguous_patterns = {
            "rice": ["Chinese", "Japanese", "Indian", "Thai", "Mexican"],
            "chicken": ["Italian", "Chinese", "Indian", "Mexican", "French"],
            "tomato": ["Italian", "Mexican", "Mediterranean", "Indian"],
            "onion": ["Italian", "Indian", "French", "Mexican", "Chinese"],
            "garlic": ["Italian", "Chinese", "Mediterranean", "French"],
            "oil": ["Italian", "Chinese", "Indian", "Mediterranean"],
            "salt": ["Universal"],
            "pepper": ["Universal"],
            "cheese": ["Italian", "French", "Mexican", "American"]
        }
        
        # Highly specific ingredients (strong cuisine indicators)
        specific_indicators = {
            "miso": ["Japanese"],
            "kimchi": ["Korean"],
            "garam masala": ["Indian"],
            "fish sauce": ["Thai", "Vietnamese"],
            "tahini": ["Middle Eastern"],
            "wasabi": ["Japanese"],
            "gochujang": ["Korean"],
            "harissa": ["North African"],
            "lemongrass": ["Thai"],
            "pancetta": ["Italian"],
            "tortilla": ["Mexican"]
        }
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            
            # Check for ambiguous ingredients
            for ambiguous, cuisines in ambiguous_patterns.items():
                if ambiguous in ingredient_lower:
                    ambiguous_ingredients.append({
                        "ingredient": ingredient,
                        "possible_cuisines": cuisines,
                        "ambiguity_level": "high" if len(cuisines) > 3 else "medium"
                    })
                    break
            
            # Check for clear indicators
            for specific, cuisines in specific_indicators.items():
                if specific in ingredient_lower:
                    clear_indicators.append({
                        "ingredient": ingredient,
                        "indicates_cuisine": cuisines,
                        "confidence": "high"
                    })
                    break
        
        # Calculate overall ambiguity
        total_ingredients = len(ingredients)
        ambiguous_count = len(ambiguous_ingredients)
        clear_count = len(clear_indicators)
        
        if total_ingredients == 0:
            ambiguity_score = 1.0
        else:
            ambiguity_score = ambiguous_count / total_ingredients
        
        clarity_score = clear_count / total_ingredients if total_ingredients > 0 else 0.0
        
        return {
            "ambiguous_ingredients": ambiguous_ingredients,
            "clear_indicators": clear_indicators,
            "ambiguity_score": ambiguity_score,
            "clarity_score": clarity_score,
            "overall_uncertainty": "high" if ambiguity_score > 0.7 else "medium" if ambiguity_score > 0.4 else "low"
        }
    
    def classify_cuisine(self, ingredients: List[str], description: str) -> CuisineClassification:
        """
        Classify cuisine type from ingredients and description.
        
        Args:
            ingredients: List of food ingredients
            description: Food description from vision analysis
            
        Returns:
            CuisineClassification object with complete analysis
        """
        try:
            logger.info(f"Classifying cuisine for: {description}")
            logger.info(f"Ingredients: {ingredients}")
            
            # Calculate keyword-based scores
            keyword_scores = self._calculate_keyword_scores(ingredients, description)
            
            # Calculate BERT-based scores if available
            bert_scores = {}
            if self.use_bert:
                # Create cuisine descriptions for BERT comparison
                cuisine_descriptions = {}
                for cuisine, data in self.cuisine_keywords.items():
                    desc_parts = []
                    desc_parts.extend(data.get("ingredients", [])[:5])
                    desc_parts.extend(data.get("dishes", [])[:3])
                    cuisine_descriptions[cuisine] = f"{cuisine} cuisine with " + ", ".join(desc_parts)
                
                text_for_bert = " ".join(ingredients + [description])
                bert_scores = self._calculate_bert_similarity(text_for_bert, cuisine_descriptions)
            
            # Combine scores (weighted average)
            combined_scores = {}
            all_cuisines = set(keyword_scores.keys()) | set(bert_scores.keys())
            
            for cuisine in all_cuisines:
                keyword_score = keyword_scores.get(cuisine, 0.0)
                bert_score = bert_scores.get(cuisine, 0.0)
                
                if bert_scores:
                    # Weight: 70% keywords, 30% BERT
                    combined_scores[cuisine] = (keyword_score * 0.7) + (bert_score * 0.3)
                else:
                    combined_scores[cuisine] = keyword_score
            
            # Calculate comprehensive uncertainty indicators
            uncertainty_indicators = self._calculate_uncertainty_indicators(ingredients, description, combined_scores)
            
            # Handle uncertainty with enhanced assessment
            uncertainty_level, base_confidence = self._handle_uncertainty(combined_scores, ingredients, description)
            
            # Apply sophisticated confidence weighting
            base_confidence = self._apply_confidence_weighting(base_confidence, uncertainty_indicators)
            
            # Detect fusion cuisine
            fusion_detected, fusion_types, fusion_description = self._detect_fusion_cuisine(
                ingredients, description, combined_scores
            )
            
            # Sort cuisines by score
            sorted_cuisines = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create cuisine results
            cuisine_results = []
            for i, (cuisine, score) in enumerate(sorted_cuisines[:5]):  # Top 5 cuisines
                # Adjust confidence based on ranking and uncertainty
                adjusted_confidence = score * base_confidence
                if i == 0:  # Primary cuisine gets full confidence
                    adjusted_confidence = min(adjusted_confidence * 1.1, 1.0)
                else:  # Secondary cuisines get reduced confidence
                    adjusted_confidence = adjusted_confidence * (0.8 - (i * 0.1))
                
                confidence_level = self._determine_confidence_level(adjusted_confidence)
                reasoning = self._generate_reasoning(cuisine, ingredients, description, score)
                
                # Add cultural context
                cultural_context = None
                if cuisine in self.cultural_context_db:
                    cultural_context = f"{cuisine} cuisine from {', '.join(self.cultural_context_db[cuisine]['regions'][:2])}"
                
                cuisine_result = CuisineResult(
                    name=cuisine,
                    confidence=adjusted_confidence,
                    confidence_level=confidence_level,
                    reasoning=reasoning,
                    cultural_context=cultural_context
                )
                cuisine_results.append(cuisine_result)
            
            # Handle case where no cuisines were identified
            if not cuisine_results:
                fallback_result = self._create_fallback_classification(
                    ingredients, description, "No strong cuisine indicators found"
                )
                cuisine_results.append(fallback_result)
            
            # Determine if multiple cuisines detected
            multiple_cuisines = len([r for r in cuisine_results if r.confidence > 0.3]) > 1
            
            # Calculate total confidence
            total_confidence = sum(r.confidence for r in cuisine_results[:3]) / 3 if cuisine_results else 0.1
            
            # Create cultural analysis
            primary_cuisine_name = cuisine_results[0].name
            cultural_analysis = {}
            if primary_cuisine_name in self.cultural_context_db:
                cultural_data = self.cultural_context_db[primary_cuisine_name]
                cultural_analysis = {
                    "primary_region": cultural_data["regions"][0] if cultural_data["regions"] else "Unknown",
                    "characteristics": cultural_data["characteristics"][:3],
                    "cooking_methods": cultural_data["cooking_methods"][:3],
                    "cultural_notes": cultural_data["cultural_notes"]
                }
            
            # Add fusion information to cultural analysis
            if fusion_detected:
                cultural_analysis["fusion_detected"] = True
                cultural_analysis["fusion_types"] = fusion_types
                cultural_analysis["fusion_description"] = fusion_description
            
            # Add comprehensive uncertainty analysis to cultural analysis
            cultural_analysis["uncertainty_analysis"] = {
                "overall_uncertainty": uncertainty_indicators["overall_uncertainty"],
                "uncertainty_category": uncertainty_indicators["uncertainty_category"],
                "recommendations": uncertainty_indicators["recommendations"],
                "should_request_more_info": uncertainty_indicators["should_request_more_info"],
                "uncertainty_factors": {
                    factor: {
                        "value": data["value"],
                        "uncertainty": data["uncertainty"]
                    }
                    for factor, data in uncertainty_indicators["uncertainty_factors"].items()
                }
            }
            
            # Create final classification
            classification = CuisineClassification(
                primary_cuisine=cuisine_results[0],
                all_cuisines=cuisine_results,
                multiple_cuisines_detected=multiple_cuisines,
                fusion_detected=fusion_detected,
                uncertainty_level=uncertainty_level,
                total_confidence=total_confidence,
                cultural_analysis=cultural_analysis
            )
            
            logger.success(f"Cuisine classification completed: {primary_cuisine_name} ({cuisine_results[0].confidence:.2f} confidence)")
            
            return classification
            
        except Exception as e:
            logger.error(f"Cuisine classification failed: {e}")
            
            # Return safe default
            unknown_result = CuisineResult(
                name="Unknown",
                confidence=0.0,
                confidence_level=CuisineConfidenceLevel.VERY_LOW,
                reasoning=f"Classification failed: {str(e)}",
                cultural_context="Unable to determine cultural context"
            )
            
            return CuisineClassification(
                primary_cuisine=unknown_result,
                all_cuisines=[unknown_result],
                multiple_cuisines_detected=False,
                fusion_detected=False,
                uncertainty_level="very_high",
                total_confidence=0.0,
                cultural_analysis={"error": str(e)}
            )
    
    def analyze_cuisine(self, vision_results: Dict) -> Dict:
        """
        Analyze cuisine from vision model results.
        
        Args:
            vision_results: Results from the vision model analysis
            
        Returns:
            Dictionary containing cuisine analysis results
        """
        try:
            # Extract ingredients and description from vision results
            ingredients = vision_results.get('ingredients', [])
            description = vision_results.get('description', 'Unknown food item')
            
            # Classify cuisine
            classification = self.classify_cuisine(ingredients, description)
            
            # Format results for API response
            results = {
                'primary_cuisine': {
                    'name': classification.primary_cuisine.name,
                    'confidence': classification.primary_cuisine.confidence,
                    'confidence_level': classification.primary_cuisine.confidence_level.value,
                    'reasoning': classification.primary_cuisine.reasoning,
                    'cultural_context': classification.primary_cuisine.cultural_context
                },
                'all_cuisines': [
                    {
                        'name': cuisine.name,
                        'confidence': cuisine.confidence,
                        'confidence_level': cuisine.confidence_level.value,
                        'reasoning': cuisine.reasoning,
                        'cultural_context': cuisine.cultural_context
                    }
                    for cuisine in classification.all_cuisines
                ],
                'multiple_cuisines_detected': classification.multiple_cuisines_detected,
                'fusion_detected': classification.fusion_detected,
                'uncertainty_level': classification.uncertainty_level,
                'total_confidence': classification.total_confidence,
                'cultural_analysis': classification.cultural_analysis,
                'analysis_status': 'success',
                'timestamp': None  # Will be set by operator layer
            }
            
            logger.success("Cuisine analysis completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Cuisine analysis failed: {e}")
            return {
                'primary_cuisine': {
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'confidence_level': 'very_low',
                    'reasoning': 'Analysis failed',
                    'cultural_context': 'Unable to determine'
                },
                'all_cuisines': [],
                'multiple_cuisines_detected': False,
                'fusion_detected': False,
                'uncertainty_level': 'very_high',
                'total_confidence': 0.0,
                'cultural_analysis': {},
                'analysis_status': 'error',
                'error_message': str(e)
            }


# Convenience function for quick cuisine analysis
def analyze_cuisine_from_vision(vision_results: Dict, use_bert: bool = True) -> Dict:
    """
    Quick utility function to analyze cuisine from vision results.
    
    Args:
        vision_results: Results from vision model analysis
        use_bert: Whether to use BERT for semantic analysis
        
    Returns:
        Dictionary containing cuisine analysis results
    """
    classifier = CuisineClassifier(use_bert=use_bert)
    return classifier.analyze_cuisine(vision_results)