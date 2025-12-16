"""
FastAPI backend components for FoodVisionAI

Contains API endpoints, request/response models, and middleware.
"""

from .main import app, create_app
from .endpoints import upload_router, status_router, analytics_router
from .middleware import LoggingMiddleware, ErrorHandlingMiddleware

__all__ = [
    "app",
    "create_app", 
    "upload_router",
    "status_router",
    "analytics_router",
    "LoggingMiddleware",
    "ErrorHandlingMiddleware"
]