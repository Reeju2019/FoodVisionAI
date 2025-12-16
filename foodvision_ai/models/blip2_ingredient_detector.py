"""
BLIP-2 Ingredient Detector for FoodVisionAI

Uses pretrained BLIP-2 model for ingredient extraction from food images.
NO TRAINING REQUIRED - Works out of the box!
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from typing import List, Dict, Optional
from loguru import logger
import re
import os


class BLIP2IngredientDetector:
    """
    Ingredient detector using BLIP-2 vision-language model.
    
    Advantages:
    - No training required
    - Works immediately
    - Understands food context
    - Can list ingredients naturally
    """
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: Optional[str] = None, use_lora: bool = False, lora_path: Optional[str] = None):
        """
        Initialize BLIP-2 ingredient detector.

        Args:
            model_name: HuggingFace model name or path to fine-tuned model (default: blip2-opt-2.7b)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            use_lora: Whether to load LoRA fine-tuned model
            lora_path: Path to LoRA weights (e.g., 'blip2_german_food_lora')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_path = lora_path

        logger.info(f"Loading BLIP-2 model: {model_name}")
        if use_lora and lora_path:
            logger.info(f"🇩🇪 Loading German food fine-tuned LoRA weights from: {lora_path}")
        logger.info(f"Device: {self.device}")

        # Check GPU availability
        if self.device == 'cuda':
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'

        try:
            # Load processor (always from base model)
            base_model_name = "Salesforce/blip2-opt-2.7b" if use_lora else model_name
            self.processor = Blip2Processor.from_pretrained(base_model_name)

            # Load base model
            if self.device == 'cuda':
                logger.info("Loading model on GPU with FP16 for faster inference...")
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            else:
                logger.info("Loading model on CPU...")
                self.model = Blip2ForConditionalGeneration.from_pretrained(base_model_name)
                self.model.to(self.device)

            # Load LoRA weights if specified
            if use_lora and lora_path:
                if os.path.exists(lora_path):
                    try:
                        from peft import PeftModel
                        logger.info(f"Loading LoRA weights from {lora_path}...")
                        self.model = PeftModel.from_pretrained(self.model, lora_path)
                        self.model = self.model.to(self.device)
                        logger.success(f"✅ LoRA weights loaded successfully!")
                    except ImportError:
                        logger.error("peft library not installed. Install with: pip install peft")
                        logger.warning("Falling back to base BLIP-2 model without fine-tuning")
                    except Exception as e:
                        logger.error(f"Failed to load LoRA weights: {e}")
                        logger.warning("Falling back to base BLIP-2 model without fine-tuning")
                else:
                    logger.warning(f"LoRA path not found: {lora_path}")
                    logger.warning("Falling back to base BLIP-2 model without fine-tuning")

            self.model.eval()
            logger.success(f"✅ BLIP-2 model loaded successfully on {self.device}")

            # Show memory usage if GPU
            if self.device == 'cuda':
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            raise
    
    def extract_ingredients(self, image: Image.Image, max_ingredients: int = 15) -> List[str]:
        """
        Extract ingredients from food image.
        
        Args:
            image: PIL Image object
            max_ingredients: Maximum number of ingredients to extract
            
        Returns:
            List of ingredient names
        """
        try:
            # Improved prompt for ingredient extraction (not dish name)
            prompt = "Question: What are the individual food items and ingredients you can see in this image? List each ingredient separately (like rice, chicken, tomato, onion, spices). Answer:"

            # Process image and prompt
            inputs = self.processor(image, prompt, return_tensors="pt")

            # Move to device
            if self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response with anti-repetition parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    min_length=10,
                    num_beams=5,
                    do_sample=False,
                    repetition_penalty=2.0,  # Penalize repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    early_stopping=True
                )
            
            # Decode response
            ingredients_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse ingredients from text
            ingredients = self._parse_ingredients(ingredients_text, max_ingredients)
            
            logger.info(f"Extracted {len(ingredients)} ingredients: {ingredients}")
            return ingredients
            
        except Exception as e:
            logger.error(f"Ingredient extraction failed: {e}")
            return []
    
    def _parse_ingredients(self, text: str, max_ingredients: int) -> List[str]:
        """
        Parse ingredient list from generated text.

        Args:
            text: Generated text from BLIP-2
            max_ingredients: Maximum number of ingredients

        Returns:
            List of cleaned ingredient names
        """
        # Remove common prefixes and the prompt text
        text = text.lower()

        # Remove the full prompt response prefix
        text = re.sub(r'the individual food items and ingredients you can see in this image are:?\s*', '', text).strip()
        text = re.sub(r'^(question:|answer:|ingredients:|the ingredients are:?|this food contains:?|the main ingredients of this food are:?)', '', text).strip()
        text = re.sub(r'what (are the individual food items and )?ingredients (can you see|you can see) in this (food|image)\?.*?answer:', '', text).strip()
        text = re.sub(r'list (the main ingredients|each ingredient separately)\.?', '', text).strip()

        # If the text is just a dish name (e.g., "chicken biryani"), try to expand it
        # Common dish names that should be expanded
        dish_to_ingredients = {
            'chicken biryani': ['rice', 'chicken', 'onions', 'tomatoes', 'yogurt', 'spices', 'herbs'],
            'biryani': ['rice', 'meat', 'onions', 'tomatoes', 'yogurt', 'spices'],
            'pizza': ['dough', 'cheese', 'tomato sauce', 'toppings'],
            'burger': ['bun', 'patty', 'lettuce', 'tomato', 'cheese', 'sauce'],
            'salad': ['lettuce', 'vegetables', 'dressing'],
            'pasta': ['pasta', 'sauce', 'cheese'],
            'sandwich': ['bread', 'filling', 'vegetables'],
        }

        # Check if the response is just a dish name
        text_clean = text.strip().lower()
        for dish_name, ingredients in dish_to_ingredients.items():
            if text_clean == dish_name or text_clean.endswith(dish_name):
                logger.warning(f"BLIP-2 returned dish name '{dish_name}' instead of ingredients. Using fallback ingredient list.")
                return ingredients[:max_ingredients]

        # Split by common delimiters
        ingredients = []

        # Try comma-separated first
        if ',' in text:
            ingredients = [ing.strip() for ing in text.split(',')]
        # Try "and" separated
        elif ' and ' in text:
            ingredients = [ing.strip() for ing in text.split(' and ')]
        # Try newline separated
        elif '\n' in text:
            ingredients = [ing.strip() for ing in text.split('\n')]
        # Try period separated
        elif '.' in text:
            ingredients = [ing.strip() for ing in text.split('.')]
        else:
            # Single ingredient or space-separated
            ingredients = [text.strip()]

        # Clean up ingredients and remove duplicates
        cleaned = []
        seen = set()

        for ing in ingredients:
            # Remove numbers, bullets, dashes at start
            ing = re.sub(r'^[\d\-\*\•\·]+\.?\s*', '', ing)
            # Remove extra whitespace
            ing = ' '.join(ing.split())
            # Remove any remaining question/answer text
            ing = re.sub(r'^(question|answer):', '', ing).strip()

            # Filter out garbled text (too many repeated characters)
            # Check for patterns like "isisisisis" or "arardarard"
            if re.search(r'(.{2,})\1{3,}', ing):  # Same 2+ chars repeated 4+ times
                continue

            # Filter out ingredients with too many repeated words
            words = ing.split()
            if len(words) > 1:
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                # Skip if any word appears more than 3 times
                if any(count > 3 for count in word_counts.values()):
                    continue

            # Filter out very long "words" (likely garbled)
            if any(len(word) > 20 for word in words):
                continue

            # Skip empty, very short, or duplicates
            if len(ing) > 2 and ing not in seen:
                # Normalize ingredient (remove duplicate words within the ingredient)
                words_unique = []
                for word in words:
                    if word not in words_unique:
                        words_unique.append(word)
                ing_normalized = ' '.join(words_unique)

                if ing_normalized not in seen:
                    cleaned.append(ing_normalized)
                    seen.add(ing_normalized)

        # Limit to max_ingredients
        return cleaned[:max_ingredients]
    
    def get_detailed_analysis(self, image: Image.Image) -> Dict:
        """
        Get detailed food analysis including ingredients and description.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with ingredients, description, and confidence
        """
        try:
            # Extract ingredients
            ingredients = self.extract_ingredients(image)
            
            # Generate general description
            desc_prompt = "Question: Describe this food dish. What is it? Answer:"
            inputs = self.processor(image, desc_prompt, return_tensors="pt")
            
            if self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100, num_beams=5)
            
            description = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "ingredients": ingredients,
                "description": description,
                "num_ingredients": len(ingredients),
                "confidence": "high" if len(ingredients) > 3 else "medium",
                "model": self.model_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {e}")
            return {
                "ingredients": [],
                "description": "Analysis failed",
                "num_ingredients": 0,
                "confidence": "low",
                "model": self.model_name,
                "status": "error",
                "error": str(e)
            }

