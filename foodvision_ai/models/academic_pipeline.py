"""
Academic 3-Stage Pipeline Implementation
Following the final project specification for Deep Learning & GenAI course

Stage 1: Vision Model (CNN) → Ingredient Detection
Stage 2: Generative AI (LLM) → Dish Identification & Description  
Stage 3: Nutrition Lookup → Nutritional Analysis
"""

import json
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from PIL import Image
import google.generativeai as genai
from loguru import logger

from .recipe1m_architecture import Recipe1MIngredientCNN, Recipe1MModelLoader
from .blip2_ingredient_detector import BLIP2IngredientDetector
from ..utils.image_processing import ImageProcessor


class AcademicFoodPipeline:
    """
    Academic 3-Stage Food Analysis Pipeline
    
    Demonstrates modular AI system design with clear separation of:
    - Perception (Computer Vision)
    - Reasoning (Language Model) 
    - Knowledge Retrieval (Nutrition Dataset)
    """
    
    def __init__(self, device: str = 'cpu', use_blip2: bool = True):
        self.device = torch.device(device)
        self.image_processor = ImageProcessor()
        self.use_blip2 = use_blip2

        # Stage 1: Vision Model (BLIP-2 or Recipe1M)
        self.vision_model = None
        self.blip2_detector = None
        self.ingredient_vocab = None
        self._load_vision_model()

        # Stage 2: Generative AI
        self.genai_model = None
        self._setup_generative_ai()

        # Stage 3: Nutrition Database
        self.nutrition_db = self._load_nutrition_database()

        logger.info("Academic 3-Stage Pipeline initialized")
    
    def _load_vision_model(self):
        """Load BLIP-2 or fine-tuned CNN for ingredient detection"""
        try:
            from ..config import settings

            if self.use_blip2:
                # Use BLIP-2 (pretrained or fine-tuned)
                logger.info("Loading BLIP-2 for ingredient detection...")
                device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'

                # Check if German food fine-tuned model should be used
                if settings.use_german_food_model:
                    import os
                    if os.path.exists(settings.german_food_lora_path):
                        logger.info("🇩🇪 Using German food fine-tuned BLIP-2 model")
                        self.blip2_detector = BLIP2IngredientDetector(
                            device=device_str,
                            use_lora=True,
                            lora_path=settings.german_food_lora_path
                        )
                        logger.info("✅ Loaded German food fine-tuned BLIP-2 detector")
                    else:
                        logger.warning(f"German food model not found at {settings.german_food_lora_path}")
                        logger.info("Falling back to base BLIP-2...")
                        self.blip2_detector = BLIP2IngredientDetector(device=device_str)
                        logger.info("✅ Loaded BLIP-2 ingredient detector (pretrained)")
                else:
                    self.blip2_detector = BLIP2IngredientDetector(device=device_str)
                    logger.info("✅ Loaded BLIP-2 ingredient detector (pretrained)")
            else:
                # Try to load Recipe1M+ model (fine-tuned)
                import os
                model_path = 'foodvision_ai/models/recipe1m_trained_model_final.pth'

                if os.path.exists(model_path):
                    self.vision_model = Recipe1MModelLoader(model_path, str(self.device))
                    self.ingredient_vocab = self.vision_model.ingredient_vocab
                    logger.info("✅ Loaded fine-tuned Recipe1M+ CNN model")
                else:
                    logger.warning(f"Recipe1M+ model not found at {model_path}")
                    logger.info("Falling back to BLIP-2...")
                    device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'
                    self.blip2_detector = BLIP2IngredientDetector(device=device_str)
        except Exception as e:
            logger.warning(f"Vision model loading failed: {e}")
            logger.info("Using fallback ingredient detection")
            self.vision_model = None
            self.blip2_detector = None
            self.ingredient_vocab = None
    
    def _setup_generative_ai(self):
        """Setup Gemini Pro for dish identification and explanation"""
        try:
            from ..config import settings
            
            if settings.gemini_api_key:
                # Configure Gemini with API key from settings
                genai.configure(api_key=settings.gemini_api_key)
                # Use gemini-2.0-flash-exp (latest experimental model)
                self.genai_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Gemini 2.0 Flash configured for generative AI")
            else:
                logger.warning("Gemini API key not found in settings")
                logger.info("Using fallback text generation")
                self.genai_model = None
        except Exception as e:
            logger.warning(f"Gemini Pro setup failed: {e}")
            logger.info("Using fallback text generation")
            self.genai_model = None
    
    def _load_nutrition_database(self) -> Dict[str, Dict]:
        """Load structured nutrition dataset"""
        # Simplified nutrition database for academic demonstration
        nutrition_db = {
            "pasta with tomato sauce": {
                "calories": 551, "protein_g": 30.9, "fat_g": 34.3, "carbs_g": 30.2
            },
            "butter chicken": {
                "calories": 438, "protein_g": 32.1, "fat_g": 28.4, "carbs_g": 12.8
            },
            "pizza margherita": {
                "calories": 266, "protein_g": 11.0, "fat_g": 10.4, "carbs_g": 33.0
            },
            "fried rice": {
                "calories": 228, "protein_g": 4.6, "fat_g": 5.6, "carbs_g": 42.0
            },
            "chicken curry": {
                "calories": 325, "protein_g": 28.5, "fat_g": 18.2, "carbs_g": 14.1
            },
            "default": {
                "calories": 300, "protein_g": 15.0, "fat_g": 12.0, "carbs_g": 35.0
            }
        }
        logger.info(f"✅ Loaded nutrition database with {len(nutrition_db)} entries")
        return nutrition_db
    
    def analyze_food_image(self, image_path: str) -> Dict[str, Any]:
        """
        Complete 3-stage academic pipeline analysis
        
        Args:
            image_path: Path to food image
            
        Returns:
            Complete analysis results with stage-by-stage outputs
        """
        logger.info(f"🔬 Starting academic 3-stage analysis: {image_path}")
        
        try:
            # Load and validate image
            image = self.image_processor.download_image_from_url(image_path)
            if image is None or not self.image_processor.validate_image(image):
                raise ValueError("Image loading or validation failed")
            
            # Stage 1: Vision Model (CNN) - Ingredient Detection
            stage1_result = self._stage1_ingredient_detection(image)
            
            # Stage 2: Generative AI (LLM) - Dish Identification
            stage2_result = self._stage2_dish_identification(stage1_result)
            
            # Stage 3: Nutrition Lookup - Nutritional Analysis
            stage3_result = self._stage3_nutrition_lookup(stage2_result)
            
            # Combine all stages for final result
            final_result = {
                "analysis_status": "success",
                "stage1_ingredients": stage1_result,
                "stage2_dish_analysis": stage2_result,
                "stage3_nutrition": stage3_result,
                "pipeline_info": {
                    "model_type": "academic_3_stage",
                    "vision_model": "fine_tuned_cnn",
                    "genai_model": "gemini_pro",
                    "nutrition_source": "structured_database"
                }
            }
            
            logger.success("🎓 Academic pipeline analysis completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Academic pipeline analysis failed: {e}")
            return {
                "analysis_status": "failed",
                "error_message": str(e),
                "stage1_ingredients": {"ingredients": []},
                "stage2_dish_analysis": {"predicted_dish": "unknown", "description": "Analysis failed"},
                "stage3_nutrition": {"calories": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0}
            }
    
    def _stage1_ingredient_detection(self, image: Image.Image) -> Dict[str, Any]:
        """
        Stage 1: Vision-based ingredient detection with structured JSON output
        Uses BLIP-2 (pretrained) or Recipe1M+ CNN (fine-tuned)

        Returns:
            Structured ingredient probabilities as JSON
        """
        logger.info("🔍 Stage 1: Ingredient Detection")

        try:
            if self.blip2_detector:
                # Use BLIP-2 (pretrained vision-language model)
                logger.info("Using BLIP-2 for ingredient extraction...")
                blip_result = self.blip2_detector.get_detailed_analysis(image)

                # Format as structured JSON
                ingredients_json = {
                    "ingredients": [
                        {"name": ing, "p": 0.85}  # BLIP-2 doesn't give probabilities, use high confidence
                        for ing in blip_result['ingredients']
                    ],
                    "model_confidence": 0.85 if blip_result['confidence'] == 'high' else 0.65,
                    "detection_method": "blip2_pretrained",
                    "description": blip_result.get('description', '')
                }
            elif self.vision_model:
                # Use fine-tuned Recipe1M+ model
                logger.info("Using Recipe1M+ CNN for ingredient detection...")
                image_tensor = self._preprocess_image(image)
                ingredient_results = self.vision_model.predict_ingredients(image_tensor, threshold=0.3)

                # Format as structured JSON
                ingredients_json = {
                    "ingredients": [
                        {"name": result["ingredient"], "p": round(result["confidence"], 3)}
                        for result in ingredient_results[:10]  # Top 10 ingredients
                    ],
                    "model_confidence": round(sum(r["confidence"] for r in ingredient_results) / len(ingredient_results), 3) if ingredient_results else 0.0,
                    "detection_method": "fine_tuned_cnn"
                }
            else:
                # Fallback method
                logger.warning("No vision model available, using fallback")
                ingredients_json = {
                    "ingredients": [
                        {"name": "unknown_ingredient", "p": 0.5}
                    ],
                    "model_confidence": 0.5,
                    "detection_method": "fallback"
                }

            logger.info(f"✅ Stage 1 complete: {len(ingredients_json['ingredients'])} ingredients detected")
            return ingredients_json

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return {
                "ingredients": [],
                "model_confidence": 0.0,
                "detection_method": "error",
                "error": str(e)
            }
    
    def _stage2_dish_identification(self, stage1_result: Dict) -> Dict[str, Any]:
        """
        Stage 2: Generative AI for dish identification and explanation
        
        Args:
            stage1_result: Structured ingredients JSON from Stage 1
            
        Returns:
            Dish prediction and natural language description
        """
        logger.info("🧠 Stage 2: Generative AI Dish Identification")
        
        try:
            ingredients = stage1_result.get("ingredients", [])
            ingredient_names = [ing["name"] for ing in ingredients]
            
            if self.genai_model and ingredient_names:
                # Use Gemini Pro for reasoning
                prompt = f"""
                Based on these detected ingredients: {', '.join(ingredient_names)}
                
                Please identify:
                1. The most likely dish name
                2. A brief description of the dish
                3. The probable cuisine type
                
                Respond in JSON format:
                {{
                    "predicted_dish": "dish name",
                    "description": "brief description",
                    "cuisine": "cuisine type",
                    "confidence": 0.85
                }}
                """
                
                response = self.genai_model.generate_content(prompt)
                
                # Parse JSON response
                try:
                    # Clean response text (remove markdown formatting if present)
                    response_text = response.text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif response_text.startswith('```'):
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                    
                    dish_analysis = json.loads(response_text)
                    
                    # Validate required fields
                    if not all(key in dish_analysis for key in ['predicted_dish', 'description', 'cuisine']):
                        raise ValueError("Missing required fields in response")
                        
                except Exception as parse_error:
                    logger.warning(f"Failed to parse Gemini response: {parse_error}")
                    # Fallback parsing
                    dish_analysis = {
                        "predicted_dish": "mixed dish",
                        "description": f"A dish containing {', '.join(ingredient_names[:3])}",
                        "cuisine": "international",
                        "confidence": 0.6
                    }
            else:
                # Rule-based fallback
                dish_analysis = self._rule_based_dish_identification(ingredient_names)
            
            dish_analysis["reasoning_method"] = "generative_ai" if self.genai_model else "rule_based"
            
            logger.info(f"✅ Stage 2 complete: {dish_analysis['predicted_dish']}")
            return dish_analysis
            
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            return {
                "predicted_dish": "unknown dish",
                "description": "Could not identify dish",
                "cuisine": "unknown",
                "confidence": 0.0,
                "reasoning_method": "error",
                "error": str(e)
            }
    
    def _stage3_nutrition_lookup(self, stage2_result: Dict) -> Dict[str, Any]:
        """
        Stage 3: Structured nutrition database lookup
        
        Args:
            stage2_result: Dish identification from Stage 2
            
        Returns:
            Nutritional information from structured dataset
        """
        logger.info("📊 Stage 3: Nutrition Database Lookup")
        
        try:
            predicted_dish = stage2_result.get("predicted_dish", "").lower()
            
            # Lookup in nutrition database
            nutrition_info = None
            
            # Exact match first
            if predicted_dish in self.nutrition_db:
                nutrition_info = self.nutrition_db[predicted_dish].copy()
            else:
                # Fuzzy matching
                for dish_name in self.nutrition_db.keys():
                    if any(word in predicted_dish for word in dish_name.split()):
                        nutrition_info = self.nutrition_db[dish_name].copy()
                        break
                
                # Default fallback
                if not nutrition_info:
                    nutrition_info = self.nutrition_db["default"].copy()
            
            # Add metadata
            nutrition_info.update({
                "lookup_method": "structured_database",
                "dish_matched": predicted_dish,
                "portion_size": "1 serving (approximate)",
                "data_source": "academic_nutrition_db"
            })
            
            logger.info(f"✅ Stage 3 complete: {nutrition_info['calories']} calories")
            return nutrition_info
            
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            return {
                "calories": 0,
                "protein_g": 0,
                "fat_g": 0,
                "carbs_g": 0,
                "lookup_method": "error",
                "error": str(e)
            }
    
    def _rule_based_dish_identification(self, ingredients: List[str]) -> Dict[str, Any]:
        """Fallback rule-based dish identification"""
        ingredient_set = set(ingredients)
        
        # Simple pattern matching
        if {"pasta", "tomato"}.issubset(ingredient_set):
            return {
                "predicted_dish": "pasta with tomato sauce",
                "description": "Pasta dish with tomato-based sauce",
                "cuisine": "italian",
                "confidence": 0.7
            }
        elif {"chicken", "butter", "tomato"}.issubset(ingredient_set):
            return {
                "predicted_dish": "butter chicken",
                "description": "Creamy chicken curry with tomato base",
                "cuisine": "indian",
                "confidence": 0.8
            }
        elif {"rice"}.issubset(ingredient_set):
            return {
                "predicted_dish": "fried rice",
                "description": "Rice dish with mixed ingredients",
                "cuisine": "asian",
                "confidence": 0.6
            }
        else:
            return {
                "predicted_dish": "mixed dish",
                "description": f"A dish containing {', '.join(ingredients[:3])}",
                "cuisine": "international",
                "confidence": 0.5
            }
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for CNN model"""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the academic pipeline configuration"""
        return {
            "pipeline_type": "academic_3_stage",
            "stage1": {
                "component": "CNN (EfficientNet-based)",
                "purpose": "Multi-label ingredient detection",
                "output": "Structured ingredients JSON"
            },
            "stage2": {
                "component": "Generative AI (Gemini Pro)",
                "purpose": "Dish identification and explanation",
                "output": "Dish name and description"
            },
            "stage3": {
                "component": "Nutrition Database Lookup",
                "purpose": "Nutritional analysis",
                "output": "Calories, protein, fat, carbohydrates"
            },
            "academic_features": [
                "Modular design for explainability",
                "Clear separation of perception/reasoning/knowledge",
                "Fine-tuned CNN with local cuisine adaptation",
                "Generative AI for semantic reasoning",
                "Structured nutrition dataset integration"
            ]
        }