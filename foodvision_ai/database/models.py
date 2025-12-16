"""
Pydantic models for database schema and data validation.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelRemarkEntry(BaseModel):
    """Individual entry in the Model_Remark log."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    component: str = Field(..., description="Component that generated the message")
    status: str = Field(..., description="Status of the operation (success, warning, error)")
    message: str = Field(..., description="Detailed message about the operation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VisionModelResult(BaseModel):
    """Vision model analysis results."""
    ingredients: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    completed: bool = Field(default=False)
    timestamp: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class NutritionModelResult(BaseModel):
    """Nutrition model analysis results."""
    calories: float = Field(default=0.0, ge=0.0)
    fat: float = Field(default=0.0, ge=0.0)
    carbohydrates: float = Field(default=0.0, ge=0.0)
    protein: float = Field(default=0.0, ge=0.0)
    portion_size: str = Field(default="")
    confidence_range: str = Field(default="")
    completed: bool = Field(default=False)
    timestamp: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class CuisineResult(BaseModel):
    """Individual cuisine classification result."""
    name: str = Field(..., description="Cuisine name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class CuisineModelResult(BaseModel):
    """Cuisine model analysis results."""
    cuisines: List[CuisineResult] = Field(default_factory=list)
    completed: bool = Field(default=False)
    timestamp: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class DescriptionJson(BaseModel):
    """Complete analysis results structure."""
    status: ProcessingStatus = Field(default=ProcessingStatus.PROCESSING)
    image_url: str = Field(default="")
    vision_model: VisionModelResult = Field(default_factory=VisionModelResult)
    nutrition_model: NutritionModelResult = Field(default_factory=NutritionModelResult)
    cuisine_model: CuisineModelResult = Field(default_factory=CuisineModelResult)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FoodAnalysisRecord(BaseModel):
    """Complete database record for food analysis."""
    id: str = Field(..., alias="_id", description="Unique image identifier")
    description_json: DescriptionJson = Field(default_factory=DescriptionJson)
    in_progress: bool = Field(default=True)
    is_error: bool = Field(default=False)
    model_remark: List[ModelRemarkEntry] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('model_remark', pre=True)
    def validate_model_remark(cls, v):
        """Ensure model_remark is always a list."""
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]


class CreateAnalysisRequest(BaseModel):
    """Request model for creating new analysis record."""
    image_id: str = Field(..., description="Unique identifier for the image")
    image_url: str = Field(..., description="Google Drive public URL for the image")


class UpdateAnalysisRequest(BaseModel):
    """Request model for updating analysis results."""
    vision_result: Optional[VisionModelResult] = None
    nutrition_result: Optional[NutritionModelResult] = None
    cuisine_result: Optional[CuisineModelResult] = None
    status: Optional[ProcessingStatus] = None
    in_progress: Optional[bool] = None
    is_error: Optional[bool] = None
    add_remark: Optional[ModelRemarkEntry] = None


class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status."""
    image_id: str
    status: ProcessingStatus
    progress: Dict[str, Dict[str, Union[bool, float]]]
    results: Dict[str, Any]
    errors: List[str]
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProgressInfo(BaseModel):
    """Progress information for individual models."""
    completed: bool = Field(default=False)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)