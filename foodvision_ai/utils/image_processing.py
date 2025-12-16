"""
Image processing utilities for FoodVisionAI

Handles image download from public URLs, preprocessing for ML models,
and basic image validation.
"""

import io
import requests
from PIL import Image
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger


class ImageProcessor:
    """Utility class for processing food images from public URLs."""
    
    def __init__(self, max_size: Tuple[int, int] = (512, 512)):
        """
        Initialize ImageProcessor with configuration.
        
        Args:
            max_size: Maximum image dimensions (width, height) for processing
        """
        self.max_size = max_size
        self.supported_formats = {'JPEG', 'PNG', 'WebP', 'JPG'}
    
    def download_image_from_url(self, url: str, timeout: int = 30) -> Optional[Image.Image]:
        """
        Download and load image from a public URL or local file path.
        
        Args:
            url: Public URL of the image or local file path
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image object if successful, None if failed
        """
        try:
            # Check if it's a local file path
            if not url.startswith(('http://', 'https://')):
                logger.info(f"Loading image from local path: {url}")
                
                # Handle relative paths
                if url.startswith('/uploads/'):
                    # Convert to absolute path
                    from pathlib import Path
                    url = str(Path.cwd() / url.lstrip('/'))
                
                # Load local image
                image = Image.open(url)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                logger.success(f"Successfully loaded local image: {image.size}")
                return image
            
            # Convert Google Drive share URL to direct download URL if needed
            if 'drive.google.com' in url and '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            logger.info(f"Downloading image from URL: {url}")
            
            # Download image with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Load image from response content
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.success(f"Successfully downloaded image: {image.size}")
            return image
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading image: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing image from URL: {e}")
            return None
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate if image is suitable for processing.
        
        Args:
            image: PIL Image object
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Check image size (minimum dimensions)
            if image.size[0] < 64 or image.size[1] < 64:
                logger.warning(f"Image too small: {image.size}")
                return False
            
            # Check image mode - should be RGB after conversion
            if image.mode not in ['RGB', 'RGBA', 'L']:
                logger.warning(f"Unsupported image mode: {image.mode}")
                return False
            
            # Basic validation - check if image has valid dimensions and data
            try:
                # Try to access image data without destroying it
                _ = image.getbbox()  # Non-destructive check
                logger.success(f"Image validation passed: {image.size}, mode: {image.mode}")
                return True
            except Exception as e:
                logger.warning(f"Image data validation failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def preprocess_for_ml(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for machine learning models.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Resize image while maintaining aspect ratio
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0, 1]
            
            logger.info(f"Preprocessed image shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def process_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """
        Complete pipeline: download, validate, and preprocess image from URL.
        
        Args:
            url: Public URL of the image
            
        Returns:
            Preprocessed image array if successful, None if failed
        """
        # Download image
        image = self.download_image_from_url(url)
        if image is None:
            return None
        
        # Validate image
        if not self.validate_image(image):
            return None
        
        # Preprocess for ML
        try:
            processed_image = self.preprocess_for_ml(image)
            return processed_image
        except Exception:
            return None


# Convenience function for quick image processing
def process_food_image_from_url(url: str, max_size: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
    """
    Quick utility function to process a food image from a public URL.
    
    Args:
        url: Public URL of the food image
        max_size: Maximum image dimensions for processing
        
    Returns:
        Preprocessed image array if successful, None if failed
    """
    processor = ImageProcessor(max_size=max_size)
    return processor.process_image_from_url(url)