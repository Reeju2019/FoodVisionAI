"""
Configuration settings for FoodVisionAI

Manages environment variables, database connections, and model configurations.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    app_name: str = "FoodVisionAI"
    debug: bool = False

    # Security settings
    cors_origins: str = "*"  # Comma-separated list of allowed origins
    allowed_hosts: str = "*"  # Comma-separated list of allowed hosts
    secret_key: str = "change-this-in-production-to-a-secure-random-key"
    
    # Database settings
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "foodvision_ai"
    
    # Google Drive settings
    google_drive_service_account_file: Optional[str] = None
    google_drive_credentials_json: Optional[str] = None
    google_drive_folder_id: Optional[str] = None
    google_drive_folder_name: str = "FoodVisionAI_Images"
    
    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/foodvision_ai.log"
    
    # Model settings
    vision_model_path: Optional[str] = None
    nutrition_model_path: Optional[str] = None
    cuisine_model_path: Optional[str] = None

    # Recipe1M Model settings (trained CNN for ingredient detection)
    recipe1m_model_dir: str = "foodvision_ai/models/recipe1m"
    recipe1m_model_path: str = "foodvision_ai/models/recipe1m/recipe1m_best_model.pth"
    recipe1m_vocab_path: str = "foodvision_ai/models/recipe1m/recipe1m_ingredient_vocab.json"
    recipe1m_deployment_info_path: str = "foodvision_ai/models/recipe1m/recipe1m_deployment_info.json"
    recipe1m_threshold: float = 0.2  # Confidence threshold for predictions
    recipe1m_top_k: int = 10  # Number of top predictions to return

    # BLIP-2 Fine-tuned Model settings (German food LoRA)
    use_german_food_model: bool = False  # Set to True to use German food fine-tuned model
    german_food_lora_path: str = "blip2_german_food_lora"  # Path to LoRA weights

    # Colab integration settings
    colab_vision_endpoint: Optional[str] = None
    colab_nutrition_endpoint: Optional[str] = None
    colab_cuisine_endpoint: Optional[str] = None
    
    # Processing settings
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    processing_timeout: int = 300  # 5 minutes
    
    # Generative AI settings (for academic pipeline)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()