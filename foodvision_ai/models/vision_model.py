"""
Vision Model for FoodVisionAI

Implements food recognition using pretrained models to extract ingredients
and generate descriptions from food images.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json
import os

from ..config import settings
from ..utils.image_processing import ImageProcessor


class VisionModel:
    """
    Vision model for food recognition and ingredient extraction.
    
    Uses a combination of:
    - EfficientNet-B0 for food classification
    - BLIP model for image captioning and ingredient extraction
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Vision Model.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Vision Model on device: {self.device}")
        
        # Initialize models
        self.food_classifier = None
        self.caption_model = None
        self.caption_processor = None
        self.image_processor = ImageProcessor(max_size=(512, 512))
        
        # Food categories mapping (Food-101 dataset classes)
        self.food_classes = self._load_food_classes()
        
        # Initialize models
        self._load_models()
    
    def _load_food_classes(self) -> List[str]:
        """Load Food-101 class names for classification."""
        # Food-101 dataset classes
        food_classes = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
            'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
            'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
            'waffles'
        ]
        return food_classes
    
    def _load_models(self):
        """Load and initialize the pretrained models."""
        try:
            # Load EfficientNet-B0 for food classification
            logger.info("Loading EfficientNet-B0 for food classification...")
            self.food_classifier = models.efficientnet_b0(pretrained=True)
            
            # Modify the classifier head for Food-101 classes
            num_classes = len(self.food_classes)
            self.food_classifier.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(self.food_classifier.classifier[1].in_features, num_classes)
            )
            
            self.food_classifier.to(self.device)
            self.food_classifier.eval()
            
            # Load BLIP model for image captioning and ingredient extraction
            logger.info("Loading BLIP-2 model for better food captioning...")
            try:
                # Try BLIP-2 first (better performance)
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
                logger.info("Successfully loaded BLIP-2 model")
            except Exception as e:
                logger.warning(f"BLIP-2 failed to load ({e}), falling back to BLIP-1")
                # Fallback to original BLIP
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
                logger.info("Successfully loaded BLIP-1 large model")
            
            self.caption_model.to(self.device)
            self.caption_model.eval()
            
            logger.success("Vision models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vision models: {e}")
            raise
    
    def _preprocess_image_for_classification(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for EfficientNet classification.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def _classify_food(self, image: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify food type using EfficientNet.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (predicted_class, confidence, top_5_predictions)
        """
        try:
            # Preprocess image
            input_tensor = self._preprocess_image_for_classification(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.food_classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            top5_predictions = [
                (self.food_classes[idx.item()], prob.item())
                for idx, prob in zip(top5_indices, top5_prob)
            ]
            
            # Get the best prediction
            best_class = self.food_classes[top5_indices[0].item()]
            best_confidence = top5_prob[0].item()
            
            logger.info(f"Food classification: {best_class} (confidence: {best_confidence:.3f})")
            
            return best_class, best_confidence, top5_predictions
            
        except Exception as e:
            logger.error(f"Food classification failed: {e}")
            return "unknown", 0.0, []
    
    def _generate_caption_and_ingredients(self, image: Image.Image) -> Tuple[str, List[str]]:
        """
        Generate image caption and extract ingredients using BLIP.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (description, ingredients_list)
        """
        try:
            # Generate food-specific caption with better prompt
            food_prompt = "This is a photo of"
            inputs = self.caption_processor(image, food_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.caption_model.generate(
                    **inputs, 
                    max_length=80, 
                    num_beams=8,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.2
                )
            
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            # Generate ingredient-focused description with better prompt
            ingredient_prompt = "The main ingredients in this dish are:"
            inputs_ingredients = self.caption_processor(
                image, 
                ingredient_prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                out_ingredients = self.caption_model.generate(
                    **inputs_ingredients, 
                    max_length=120, 
                    num_beams=8,
                    temperature=0.8,
                    do_sample=True,
                    repetition_penalty=1.3
                )
            
            ingredients_text = self.caption_processor.decode(out_ingredients[0], skip_special_tokens=True)
            
            # Extract ingredients from the generated text
            ingredients = self._extract_ingredients_from_text(ingredients_text, caption)
            
            logger.info(f"Generated caption: {caption}")
            logger.info(f"Extracted ingredients: {ingredients}")
            
            return caption, ingredients
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "A food dish", []
    
    def _extract_ingredients_from_text(self, ingredients_text: str, caption: str) -> List[str]:
        """
        Enhanced ingredient extraction with food-specific rules and dish recognition.
        
        Args:
            ingredients_text: Text generated with ingredient prompt
            caption: General image caption
            
        Returns:
            List of extracted ingredients
        """
        # Enhanced ingredient database with variations
        ingredient_database = {
            # Proteins
            'chicken': ['chicken', 'poultry', 'fowl'],
            'beef': ['beef', 'steak', 'meat'],
            'pork': ['pork', 'bacon', 'ham'],
            'fish': ['fish', 'salmon', 'tuna', 'cod'],
            'shrimp': ['shrimp', 'prawn'],
            'egg': ['egg', 'eggs'],
            
            # Dairy
            'butter': ['butter', 'ghee'],
            'cream': ['cream', 'heavy cream', 'whipping cream'],
            'cheese': ['cheese', 'cheddar', 'mozzarella', 'parmesan'],
            'milk': ['milk', 'dairy'],
            'yogurt': ['yogurt', 'curd'],
            
            # Vegetables
            'tomato': ['tomato', 'tomatoes'],
            'onion': ['onion', 'onions'],
            'garlic': ['garlic'],
            'ginger': ['ginger'],
            'pepper': ['pepper', 'bell pepper', 'capsicum'],
            'chili': ['chili', 'chilli', 'hot pepper'],
            'carrot': ['carrot', 'carrots'],
            'potato': ['potato', 'potatoes'],
            'spinach': ['spinach', 'greens'],
            'mushroom': ['mushroom', 'mushrooms'],
            
            # Grains & Starches
            'rice': ['rice', 'basmati', 'jasmine rice'],
            'pasta': ['pasta', 'spaghetti', 'noodles'],
            'bread': ['bread', 'naan', 'roti'],
            'flour': ['flour', 'wheat'],
            
            # Spices & Herbs
            'cumin': ['cumin', 'jeera'],
            'coriander': ['coriander', 'cilantro'],
            'turmeric': ['turmeric', 'haldi'],
            'garam masala': ['garam masala', 'spices'],
            'bay leaves': ['bay leaves'],
            'cardamom': ['cardamom'],
            'cinnamon': ['cinnamon'],
            
            # Oils & Fats
            'oil': ['oil', 'cooking oil', 'vegetable oil'],
            'coconut oil': ['coconut oil'],
            'olive oil': ['olive oil'],
        }
        
        # Food-specific dish recognition patterns
        dish_patterns = {
            'butter chicken': ['butter', 'chicken', 'tomato', 'cream', 'onion', 'garlic', 'ginger', 'garam masala', 'turmeric'],
            'chicken curry': ['chicken', 'onion', 'tomato', 'garlic', 'ginger', 'cumin', 'coriander', 'turmeric'],
            'biryani': ['rice', 'chicken', 'onion', 'yogurt', 'garam masala', 'saffron'],
            'pasta': ['pasta', 'tomato', 'garlic', 'olive oil', 'cheese'],
            'pizza': ['bread', 'tomato', 'cheese', 'oil'],
            'fried rice': ['rice', 'egg', 'soy sauce', 'vegetables'],
        }
        
        # Combine both texts for analysis
        combined_text = f"{ingredients_text} {caption}".lower()
        
        # First, try to match known dish patterns
        found_ingredients = []
        for dish_name, dish_ingredients in dish_patterns.items():
            if any(word in combined_text for word in dish_name.split()):
                logger.info(f"Detected dish pattern: {dish_name}")
                found_ingredients.extend(dish_ingredients)
                break
        
        # Then, extract ingredients using enhanced database
        for base_ingredient, variations in ingredient_database.items():
            for variation in variations:
                if variation in combined_text and base_ingredient not in found_ingredients:
                    found_ingredients.append(base_ingredient)
                    break
        
        # Color-based ingredient detection
        color_ingredients = {
            'red': ['tomato', 'chili', 'bell pepper'],
            'orange': ['carrot', 'turmeric'],
            'green': ['spinach', 'coriander', 'pepper'],
            'white': ['onion', 'garlic', 'cream', 'rice'],
            'yellow': ['turmeric', 'butter', 'egg'],
            'brown': ['chicken', 'beef', 'mushroom']
        }
        
        for color, ingredients in color_ingredients.items():
            if color in combined_text:
                for ingredient in ingredients:
                    if ingredient not in found_ingredients:
                        found_ingredients.append(ingredient)
        
        # If still no ingredients found, use fallback extraction
        if not found_ingredients:
            words = combined_text.replace(',', ' ').replace('.', ' ').split()
            food_words = [word for word in words if len(word) > 3 and word.isalpha()]
            found_ingredients = food_words[:5]
        
        # Remove duplicates and limit
        unique_ingredients = list(dict.fromkeys(found_ingredients))[:12]
        
        logger.info(f"Enhanced ingredient extraction found: {unique_ingredients}")
        return unique_ingredients

def _preprocess_for_recipe1m(self, image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for Recipe1M+ model inference
    """
    if self.recipe1m_model and self.recipe1m_model.preprocessing:
        preprocessing = self.recipe1m_model.preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing['mean'], 
                std=preprocessing['std']
            )
        ])
    else:
        # Default preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image_tensor = transform(image).unsqueeze(0).to(self.device)
    return image_tensor

def _generate_description_with_blip(self, image: Image.Image) -> str:
    """
    Generate food description using BLIP model
    """
    try:
        # Generate food-specific caption with better prompt
        food_prompt = "This is a photo of"
        inputs = self.caption_processor(image, food_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.caption_model.generate(
                **inputs, 
                max_length=80, 
                num_beams=8,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        description = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return description
        
    except Exception as e:
        logger.error(f"BLIP description generation failed: {e}")
        return "A food dish"

def _infer_food_type_from_ingredients(self, ingredients: List[str]) -> str:
    """
    Infer food type from detected ingredients using pattern matching
    """
    ingredient_set = set(ingredients)
    
    # Food type patterns
    patterns = {
        'butter chicken': {'butter', 'chicken', 'tomato', 'cream'},
        'chicken curry': {'chicken', 'curry', 'onion', 'garlic'},
        'pasta': {'pasta', 'tomato', 'cheese'},
        'pizza': {'cheese', 'tomato', 'bread'},
        'fried rice': {'rice', 'egg', 'soy sauce'},
        'salad': {'lettuce', 'vegetables', 'greens'},
        'soup': {'broth', 'vegetables', 'liquid'},
        'sandwich': {'bread', 'meat', 'vegetables'},
        'stir fry': {'vegetables', 'oil', 'garlic'},
        'biryani': {'rice', 'chicken', 'spices'}
    }
    
    # Find best matching pattern
    best_match = 'mixed dish'
    best_score = 0
    
    for food_type, pattern in patterns.items():
        # Calculate overlap score
        overlap = len(pattern.intersection(ingredient_set))
        score = overlap / len(pattern) if pattern else 0
        
        if score > best_score and score > 0.5:  # At least 50% match
            best_match = food_type
            best_score = score
    
    return best_match

def _get_confidence_level(self, confidence: float) -> str:
    """
    Convert numerical confidence to descriptive level
    """
    if confidence >= 0.8:
        return 'very_high'
    elif confidence >= 0.6:
        return 'high'
    elif confidence >= 0.4:
        return 'medium'
    elif confidence >= 0.2:
        return 'low'
    else:
        return 'very_low'
    
    def _detect_multiple_foods(self, image: Image.Image, top5_predictions: List[Tuple[str, float]]) -> Dict:
        """
        Enhanced multi-food detection with confidence analysis.
        
        Args:
            image: PIL Image object
            top5_predictions: Top 5 classification predictions
            
        Returns:
            Dictionary with multi-food analysis results
        """
        try:
            # Analyze confidence distribution
            confidences = [pred[1] for pred in top5_predictions]
            max_confidence = max(confidences) if confidences else 0.0
            confidence_spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0.0
            
            # Multi-food indicators
            multiple_foods_detected = False
            food_items = []
            
            # If confidence is low or spread is small, likely multiple foods
            if max_confidence < 0.6 or confidence_spread < 0.3:
                multiple_foods_detected = True
                # Include top predictions as potential food items
                for food_type, confidence in top5_predictions[:3]:
                    if confidence > 0.1:  # Minimum threshold
                        food_items.append({
                            "food_type": food_type.replace('_', ' ').title(),
                            "confidence": confidence,
                            "likelihood": "high" if confidence > 0.4 else "medium" if confidence > 0.2 else "low"
                        })
            else:
                # Single dominant food item
                food_items.append({
                    "food_type": top5_predictions[0][0].replace('_', ' ').title(),
                    "confidence": top5_predictions[0][1],
                    "likelihood": "high"
                })
            
            return {
                "multiple_foods_detected": multiple_foods_detected,
                "food_items": food_items,
                "confidence_analysis": {
                    "max_confidence": max_confidence,
                    "confidence_spread": confidence_spread,
                    "certainty_level": "high" if max_confidence > 0.7 else "medium" if max_confidence > 0.4 else "low"
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-food detection failed: {e}")
            return {
                "multiple_foods_detected": False,
                "food_items": [],
                "confidence_analysis": {
                    "max_confidence": 0.0,
                    "confidence_spread": 0.0,
                    "certainty_level": "low"
                }
            }
    
    def _calculate_overall_confidence(self, classification_confidence: float, caption_quality: str) -> Dict:
        """
        Calculate overall confidence score for the analysis.
        
        Args:
            classification_confidence: Confidence from food classification
            caption_quality: Quality assessment of generated caption
            
        Returns:
            Dictionary with confidence metrics
        """
        # Base confidence from classification
        base_confidence = classification_confidence
        
        # Adjust based on caption quality (simple heuristic)
        caption_multiplier = 1.0
        if caption_quality == "high":
            caption_multiplier = 1.1
        elif caption_quality == "low":
            caption_multiplier = 0.9
        
        # Calculate overall confidence
        overall_confidence = min(base_confidence * caption_multiplier, 1.0)
        
        # Determine confidence level
        if overall_confidence > 0.8:
            confidence_level = "very_high"
        elif overall_confidence > 0.6:
            confidence_level = "high"
        elif overall_confidence > 0.4:
            confidence_level = "medium"
        elif overall_confidence > 0.2:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level,
            "classification_confidence": classification_confidence,
            "caption_quality_factor": caption_multiplier
        }
    
    def analyze_image(self, image_url: str) -> Dict:
        """
        Complete analysis of a food image from URL.
        
        Args:
            image_url: Public URL of the food image
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting vision analysis for image: {image_url}")
            
            # Download and preprocess image
            image = self.image_processor.download_image_from_url(image_url)
            if image is None:
                raise ValueError("Failed to download or process image")
            
            # Validate image
            if not self.image_processor.validate_image(image):
                raise ValueError("Image validation failed")
            
            # Classify food type
            food_class, classification_confidence, top5_predictions = self._classify_food(image)
            
            # Generate caption and extract ingredients
            description, ingredients = self._generate_caption_and_ingredients(image)
            
            # Enhanced multi-food detection
            multi_food_analysis = self._detect_multiple_foods(image, top5_predictions)
            
            # Calculate overall confidence
            caption_quality = "high" if len(description) > 20 else "medium" if len(description) > 10 else "low"
            confidence_metrics = self._calculate_overall_confidence(classification_confidence, caption_quality)
            
            # Prepare enhanced results
            results = {
                "ingredients": ingredients,
                "description": description,
                "food_type": food_class.replace('_', ' ').title(),
                "confidence": confidence_metrics["overall_confidence"],
                "confidence_level": confidence_metrics["confidence_level"],
                "top_predictions": [
                    {
                        "food_type": pred[0].replace('_', ' ').title(),
                        "confidence": pred[1]
                    }
                    for pred in top5_predictions
                ],
                "multiple_foods": multi_food_analysis["multiple_foods_detected"],
                "multi_food_analysis": multi_food_analysis,
                "confidence_metrics": confidence_metrics,
                "analysis_status": "success",
                "timestamp": logger._get_now().isoformat() if hasattr(logger, '_get_now') else None
            }
            
            logger.success(f"Vision analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {
                "ingredients": [],
                "description": "Analysis failed",
                "food_type": "Unknown",
                "confidence": 0.0,
                "confidence_level": "very_low",
                "top_predictions": [],
                "multiple_foods": False,
                "multi_food_analysis": {
                    "multiple_foods_detected": False,
                    "food_items": [],
                    "confidence_analysis": {"max_confidence": 0.0, "confidence_spread": 0.0, "certainty_level": "low"}
                },
                "confidence_metrics": {
                    "overall_confidence": 0.0,
                    "confidence_level": "very_low",
                    "classification_confidence": 0.0,
                    "caption_quality_factor": 1.0
                },
                "analysis_status": "error",
                "error_message": str(e)
            }
    
    def analyze_image_from_pil(self, image: Image.Image) -> Dict:
        """
        Analyze a PIL Image object directly.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info("Starting vision analysis for PIL image")
            
            # Validate image
            if not self.image_processor.validate_image(image):
                raise ValueError("Image validation failed")
            
            # Classify food type
            food_class, classification_confidence, top5_predictions = self._classify_food(image)
            
            # Generate caption and extract ingredients
            description, ingredients = self._generate_caption_and_ingredients(image)
            
            # Enhanced multi-food detection
            multi_food_analysis = self._detect_multiple_foods(image, top5_predictions)
            
            # Calculate overall confidence
            caption_quality = "high" if len(description) > 20 else "medium" if len(description) > 10 else "low"
            confidence_metrics = self._calculate_overall_confidence(classification_confidence, caption_quality)
            
            # Prepare enhanced results
            results = {
                "ingredients": ingredients,
                "description": description,
                "food_type": food_class.replace('_', ' ').title(),
                "confidence": confidence_metrics["overall_confidence"],
                "confidence_level": confidence_metrics["confidence_level"],
                "top_predictions": [
                    {
                        "food_type": pred[0].replace('_', ' ').title(),
                        "confidence": pred[1]
                    }
                    for pred in top5_predictions
                ],
                "multiple_foods": multi_food_analysis["multiple_foods_detected"],
                "multi_food_analysis": multi_food_analysis,
                "confidence_metrics": confidence_metrics,
                "analysis_status": "success"
            }
            
            logger.success("Vision analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {
                "ingredients": [],
                "description": "Analysis failed",
                "food_type": "Unknown",
                "confidence": 0.0,
                "confidence_level": "very_low",
                "top_predictions": [],
                "multiple_foods": False,
                "multi_food_analysis": {
                    "multiple_foods_detected": False,
                    "food_items": [],
                    "confidence_analysis": {"max_confidence": 0.0, "confidence_spread": 0.0, "certainty_level": "low"}
                },
                "confidence_metrics": {
                    "overall_confidence": 0.0,
                    "confidence_level": "very_low",
                    "classification_confidence": 0.0,
                    "caption_quality_factor": 1.0
                },
                "analysis_status": "error",
                "error_message": str(e)
            }


# Convenience function for quick vision analysis
def analyze_food_image(image_url: str, device: Optional[str] = None) -> Dict:
    """
    Quick utility function to analyze a food image from URL.
    
    Args:
        image_url: Public URL of the food image
        device: Device to run the model on
        
    Returns:
        Dictionary containing vision analysis results
    """
    model = VisionModel(device=device)
    return model.analyze_image(image_url)