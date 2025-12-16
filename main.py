"""
Main entry point for FoodVisionAI application

Starts the FastAPI server and initializes all components.
"""

import uvicorn
from foodvision_ai.utils.logging_config import setup_logging
from foodvision_ai.config import settings


def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    
    # Start the FastAPI server
    uvicorn.run(
        "foodvision_ai.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()