"""
Recipe1M+ Custom CNN Architecture for Local Integration

This file contains the exact same model architecture as trained in Colab.
Must match exactly for loading trained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Any


class Recipe1MIngredientCNN(nn.Module):
    """
    Custom CNN for ingredient detection trained on Recipe1M+
    - Multi-label classification for 300+ ingredients
    - EfficientNet backbone with custom head
    - Attention mechanism for ingredient localization
    """
    def __init__(self, num_ingredients=300, backbone='efficientnet_b3'):
        super(Recipe1MIngredientCNN, self).__init__()
        
        self.num_ingredients = num_ingredients
        self.backbone_name = backbone
        
        # Efficient backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=True, 
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        if 'efficientnet_b0' in backbone:
            self.feature_dim = 1280
        elif 'efficientnet_b3' in backbone:
            self.feature_dim = 1536
        else:
            self.feature_dim = 1280  # Default
        
        # Multi-head attention for ingredient localization
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Ingredient classification head
        self.ingredient_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_ingredients),
            nn.Sigmoid()  # Multi-label classification
        )
        
        # Cultural context classifier (bonus feature)
        self.culture_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 20),  # 20 major cuisines
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [batch, feature_dim]
        
        # Apply attention (reshape for attention)
        features_reshaped = features.unsqueeze(1)  # [batch, 1, feature_dim]
        attended_features, attention_weights = self.attention(
            features_reshaped, features_reshaped, features_reshaped
        )
        attended_features = attended_features.squeeze(1)  # [batch, feature_dim]
        
        # Predict ingredients
        ingredients = self.ingredient_classifier(attended_features)
        
        # Predict cultural context
        culture = self.culture_classifier(features)
        
        return {
            'ingredients': ingredients,
            'culture': culture,
            'features': features,
            'attention_weights': attention_weights
        }


class Recipe1MModelLoader:
    """
    Utility class for loading and using Recipe1M+ trained model
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.ingredient_vocab = None
        self.reverse_vocab = None
        self.preprocessing = None
        self.model_config = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            self.model_config = checkpoint['model_config']
            self.ingredient_vocab = checkpoint['ingredient_vocab']
            self.reverse_vocab = checkpoint['reverse_vocab']
            self.preprocessing = checkpoint['preprocessing']
            
            # Initialize model with correct configuration
            self.model = Recipe1MIngredientCNN(
                num_ingredients=self.model_config['num_ingredients'],
                backbone=self.model_config['backbone']
            )
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Print model info
            training_stats = checkpoint.get('training_stats', {})
            print(f"✅ Recipe1M+ model loaded successfully!")
            print(f"📊 Model accuracy: {training_stats.get('final_val_accuracy', 0):.1%}")
            print(f"🥘 Ingredients: {len(self.ingredient_vocab)}")
            print(f"🧠 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Failed to load Recipe1M+ model: {e}")
            raise
    
    def predict_ingredients(self, image_tensor: torch.Tensor, threshold: float = 0.3):
        """
        Predict ingredients from image tensor
        
        Args:
            image_tensor: Preprocessed image tensor [1, 3, 224, 224]
            threshold: Confidence threshold for ingredient detection
            
        Returns:
            List of detected ingredients with confidence scores
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.model(image_tensor)
            ingredient_probs = outputs['ingredients'].cpu().numpy()[0]  # [num_ingredients]
            
            # Get ingredients above threshold
            detected_ingredients = []
            for idx, prob in enumerate(ingredient_probs):
                if prob > threshold:
                    ingredient_name = self.reverse_vocab[idx]
                    detected_ingredients.append({
                        'ingredient': ingredient_name,
                        'confidence': float(prob)
                    })
            
            # Sort by confidence
            detected_ingredients.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detected_ingredients
    
    def predict_culture(self, image_tensor: torch.Tensor):
        """
        Predict cultural context from image tensor
        
        Args:
            image_tensor: Preprocessed image tensor [1, 3, 224, 224]
            
        Returns:
            Dictionary with culture prediction and confidence
        """
        culture_names = [
            'american', 'italian', 'chinese', 'indian', 'mexican',
            'french', 'japanese', 'thai', 'korean', 'mediterranean',
            'other'
        ]
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            culture_probs = outputs['culture'].cpu().numpy()[0]
            
            # Get top culture
            top_idx = culture_probs.argmax()
            top_culture = culture_names[top_idx] if top_idx < len(culture_names) else 'other'
            top_confidence = float(culture_probs[top_idx])
            
            return {
                'culture': top_culture,
                'confidence': top_confidence,
                'all_cultures': [
                    {'culture': culture_names[i], 'confidence': float(culture_probs[i])}
                    for i in range(min(len(culture_names), len(culture_probs)))
                ]
            }
    
    def get_model_info(self):
        """Get model information"""
        return {
            'architecture': self.model_config['architecture'],
            'num_ingredients': self.model_config['num_ingredients'],
            'backbone': self.model_config['backbone'],
            'input_size': self.preprocessing['input_size'],
            'vocab_size': len(self.ingredient_vocab)
        }