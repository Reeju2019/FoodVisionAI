"""
Recipe1M Model Loader for Local Deployment

This module provides utilities to load the trained Recipe1M model
from the Colab training notebook into the local FoodVisionAI project.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import timm
from PIL import Image
from torchvision import transforms


class Recipe1MIngredientCNN(nn.Module):
    """EfficientNet backbone + calibrated MLP head for multi-label ingredient detection.
    
    This is the EXACT same architecture used in training.
    """
    
    def __init__(self, num_ingredients=500, backbone='efficientnet_b3', avg_pos_per_sample=6.52):
        super().__init__()
        
        self.num_ingredients = num_ingredients
        self.backbone_name = backbone
        
        # Load pretrained EfficientNet backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        # Feature dimension for EfficientNet-B3
        if 'efficientnet_b3' in backbone:
            self.feature_dim = 1536
        else:
            self.feature_dim = 1280
        
        # MLP classifier head
        self.ingredient_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_ingredients)
        )
        
        # Calibrated bias initialization (same as training)
        final_linear = self.ingredient_classifier[-1]
        if final_linear.bias is not None:
            p = avg_pos_per_sample / num_ingredients
            bias = float(torch.log(torch.tensor(p / (1 - p))))
            nn.init.constant_(final_linear.bias, bias)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.ingredient_classifier(features)
        return {'ingredients': logits}


class Recipe1MModelLoader:
    """Utility class to load and use the trained Recipe1M model."""
    
    def __init__(self, model_dir: str = "foodvision_ai/models/recipe1m"):
        """Initialize the model loader.
        
        Args:
            model_dir: Directory containing the model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.ingredient_vocab = None
        self.idx_to_ingredient = None
        self.deployment_info = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> Tuple[nn.Module, Dict[str, int], Dict]:
        """Load the trained model, vocabulary, and deployment info.
        
        Returns:
            Tuple of (model, ingredient_vocab, deployment_info)
        """
        # Load deployment info
        deployment_info_path = self.model_dir / "recipe1m_deployment_info.json"
        with open(deployment_info_path, 'r') as f:
            self.deployment_info = json.load(f)
        
        # Load ingredient vocabulary
        vocab_path = self.model_dir / "recipe1m_ingredient_vocab.json"
        with open(vocab_path, 'r') as f:
            self.ingredient_vocab = json.load(f)
        
        # Create reverse mapping (index -> ingredient name)
        self.idx_to_ingredient = {v: k for k, v in self.ingredient_vocab.items()}
        
        # Create model with same architecture as training
        arch_config = self.deployment_info['model_architecture']
        self.model = Recipe1MIngredientCNN(
            num_ingredients=arch_config['num_ingredients'],
            backbone=arch_config['backbone'],
            avg_pos_per_sample=arch_config['avg_pos_per_sample']
        )
        
        # Load trained weights
        checkpoint_path = self.model_dir / "recipe1m_best_model.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Ingredients: {len(self.ingredient_vocab)}")
        print(f"   Best Val F1: {self.deployment_info['training_info']['best_val_f1']:.4f}")
        
        return self.model, self.ingredient_vocab, self.deployment_info
    
    def get_transform(self):
        """Get the image preprocessing transform (same as training)."""
        norm_config = self.deployment_info['inference_config']['normalization']
        image_size = self.deployment_info['inference_config']['image_size']
        
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_config['mean'], std=norm_config['std'])
        ])
    
    def predict(self, image: Image.Image, threshold: Optional[float] = None, 
                top_k: Optional[int] = None) -> Dict:
        """Predict ingredients from a food image.
        
        Args:
            image: PIL Image of food
            threshold: Confidence threshold (default from deployment_info)
            top_k: Number of top predictions to return (default from deployment_info)
        
        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use default values from deployment info if not specified
        if threshold is None:
            threshold = self.deployment_info['inference_config']['recommended_threshold']
        if top_k is None:
            top_k = self.deployment_info['inference_config']['top_k']
        
        # Preprocess image
        transform = self.get_transform()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs['ingredients']
            probs = torch.sigmoid(logits).squeeze(0)  # [num_ingredients]
        
        # Get top-K predictions
        top_probs, top_indices = torch.topk(probs, k=top_k)
        top_ingredients = [self.idx_to_ingredient[idx.item()] for idx in top_indices]
        
        # Get threshold-based predictions
        threshold_mask = probs >= threshold
        threshold_indices = torch.where(threshold_mask)[0]
        threshold_ingredients = [self.idx_to_ingredient[idx.item()] for idx in threshold_indices]
        threshold_probs = [probs[idx].item() for idx in threshold_indices]
        
        return {
            'top_k_predictions': {
                'ingredients': top_ingredients,
                'probabilities': top_probs.cpu().tolist()
            },
            'threshold_predictions': {
                'ingredients': threshold_ingredients,
                'probabilities': threshold_probs,
                'threshold': threshold
            },
            'metadata': {
                'num_predictions_top_k': len(top_ingredients),
                'num_predictions_threshold': len(threshold_ingredients),
                'device': str(self.device)
            }
        }


# Convenience function for quick loading
def load_recipe1m_model(model_dir: str = "foodvision_ai/models/recipe1m"):
    """Quick function to load the Recipe1M model.
    
    Args:
        model_dir: Directory containing model files
    
    Returns:
        Recipe1MModelLoader instance with loaded model
    """
    loader = Recipe1MModelLoader(model_dir)
    loader.load_model()
    return loader

