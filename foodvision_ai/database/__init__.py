"""
Database operations and models for FoodVisionAI

Contains MongoDB connection utilities, data models, and CRUD operations.
"""

from .connection import DatabaseConnection, DatabaseSettings, db_connection, get_database
from .models import (
    FoodAnalysisRecord,
    CreateAnalysisRequest,
    UpdateAnalysisRequest,
    AnalysisStatusResponse,
    ModelRemarkEntry,
    VisionModelResult,
    NutritionModelResult,
    CuisineModelResult,
    ProcessingStatus,
    ProgressInfo
)
from .operations import DatabaseOperations

__all__ = [
    "DatabaseConnection",
    "DatabaseSettings", 
    "db_connection",
    "get_database",
    "FoodAnalysisRecord",
    "CreateAnalysisRequest",
    "UpdateAnalysisRequest",
    "AnalysisStatusResponse",
    "ModelRemarkEntry",
    "VisionModelResult",
    "NutritionModelResult",
    "CuisineModelResult",
    "ProcessingStatus",
    "ProgressInfo",
    "DatabaseOperations"
]