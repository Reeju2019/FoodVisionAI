"""
Enhanced Vision Model Integration

This file contains the integration code to use Recipe1M+ model in your existing system.
Add these methods to your existing VisionModel class.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List
from loguru import logger
from PIL import Image

from .recipe1m_architecture import Recipe1MModelLoader


class EnhancedVisionMixin:
    """
    Mixin class to add Recipe1M+ functionality to existing VisionModel
    """
    
    def _try_load_recipe1m_model(self):
        """
        Try to load Recipe1M+ trained model for superior ingredient detection
        """
        recipe1m_model_path = 'foodvision_ai/models/recipe1m_trained_model_final.pth'
        
        if os.path.exists(recipe1m_model_path):
            try:
                logger.info("Loading Recipe1M+ trained model...")
                self.recipe1m_model = Recipe1MModelLoader(recipe1m_model_path, self.device)
                self.use_recipe1m = True
                logger.success("✅ Recipe1M+ model loaded! Using advanced ingredient detection.")
            except Exception as e:
                logger.warning(f"Failed to load Recipe1M+ model: {e}")
                logger.info("Falling back to standard models...")
                self.use_recipe1m = False
                self.recipe1m_model = None
        else:
            logger.info("Recipe1M+ model not found. Using standard models.")
            logger.info(f"To use Recipe1M+ model, place trained model at: {recipe1m_model_path}")
            self.use_recipe1m = False
            self.recipe1m_model = None
    
    def _analyze_with_recipe1m(self, image: Image.Image, image_url: str) -> Dict:
        """
        Analyze image using Recipe1M+ trained model for superior accuracy
        """
        try:
            # Preprocess image for Recipe1M+ model
            image_tensor = self._preprocess_for_recipe1m(image)
            
            # Get ingredient predictions
            ingredient_results = self.recipe1m_model.predict_ingredients(image_tensor, threshold=0.3)
            ingredients = [result['ingredient'] for result in ingredient_results]
            
            # Get cultural context
            culture_results = self.recipe1m_model.predict_culture(image_tensor)
            
            # Calculate overall confidence
            if ingredient_results:
                avg_confidence = np.mean([result['confidence'] for result in ingredient_results])
                confidence_level = self._get_confidence_level(avg_confidence)
            else:
                avg_confidence = 0.1
                confidence_level = 'very_low'
            
            # Generate description using BLIP (as secondary source)
            description = self._generate_description_with_blip(image)
            
            # Determine food type from ingredients
            food_type = self._infer_food_type_from_ingredients(ingredients)
            
            logger.success(f"Recipe1M+ analysis completed: {len(ingredients)} ingredients detected")
            
            return {
                'analysis_status': 'success',
                'ingredients': ingredients,
                'description': description,
                'food_type': food_type,
                'confidence': float(avg_confidence),
                'confidence_level': confidence_level,
                'multiple_foods': len(ingredients) > 8,  # Heuristic for multiple foods
                'cultural_context': culture_results,
                'ingredient_details': ingredient_results,
                'model_source': 'recipe1m_trained',
                'model_info': self.recipe1m_model.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Recipe1M+ analysis failed: {e}")
            # Fallback to standard analysis
            return self._analyze_with_fallback(image, image_url)
    
    def _analyze_with_fallback(self, image: Image.Image, image_url: str) -> Dict:
        """
        Fallback analysis using standard EfficientNet + BLIP models
        """
        try:
            # Classify food type
            food_class, classification_confidence, top5_predictions = self._classify_food(image)
            
            # Generate caption and extract ingredients
            description, ingredients = self._generate_caption_and_ingredients(image)
            
            # Enhanced multi-food detection
            multi_food_analysis = self._detect_multiple_foods(image, top5_predictions)
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(classification_confidence)
            
            logger.info(f"Fallback analysis completed: {food_class} ({classification_confidence:.3f})")
            
            return {
                'analysis_status': 'success',
                'ingredients': ingredients,
                'description': description,
                'food_type': food_class.replace('_', ' ').title(),
                'confidence': float(classification_confidence),
                'confidence_level': confidence_level,
                'multiple_foods': multi_food_analysis['multiple_foods_detected'],
                'multi_food_details': multi_food_analysis,
                'top5_predictions': top5_predictions,
                'model_source': 'fallback_standard'
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return {
                'analysis_status': 'failed',
                'error_message': str(e),
                'ingredients': [],
                'description': 'Analysis failed',
                'food_type': 'unknown',
                'confidence': 0.0,
                'confidence_level': 'very_low',
                'multiple_foods': False,
                'model_source': 'error'
            }
    
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


# Integration instructions for your existing VisionModel class:
"""
To integrate Recipe1M+ into your existing VisionModel:

1. Add this import to your vision_model.py:
   from .enhanced_vision_integration import EnhancedVisionMixin

2. Modify your VisionModel class declaration:
   class VisionModel(EnhancedVisionMixin):

3. Add these lines to your __init__ method:
   # Try to load Recipe1M+ model first
   self.recipe1m_model = None
   self.use_recipe1m = False
   self._try_load_recipe1m_model()

4. Modify your analyze_image method to use the enhanced analysis:
   def analyze_image(self, image_url: str) -> Dict:
       try:
           # ... existing preprocessing code ...
           
           # Use Recipe1M+ model if available, otherwise fallback
           if self.use_recipe1m:
               return self._analyze_with_recipe1m(image, image_url)
           else:
               return self._analyze_with_fallback(image, image_url)
       except Exception as e:
           # ... existing error handling ...

That's it! Your existing code will now automatically use Recipe1M+ if available,
or fallback to your current models if not.
"""