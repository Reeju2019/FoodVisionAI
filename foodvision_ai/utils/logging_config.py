"""
Logging configuration for FoodVisionAI

Sets up structured logging with appropriate levels and formatting.
"""

import sys
from loguru import logger
from foodvision_ai.config import settings


def setup_logging():
    """Configure logging for the application."""
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with appropriate level
    log_level = "DEBUG" if settings.debug else "INFO"
    
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # Add file logger for persistent logging
    logger.add(
        "logs/foodvision_ai.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    logger.info("Logging configured successfully")