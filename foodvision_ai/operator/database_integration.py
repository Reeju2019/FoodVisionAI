"""
Database Integration for Operator Layer

Connects the Operator Layer to database for storing results and
integrates model pipeline with FastAPI endpoints.
"""
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime

from ..database import DatabaseOperations, get_database
from ..database.models import (
    UpdateAnalysisRequest,
    ModelRemarkEntry,
    VisionModelResult,
    NutritionModelResult,
    CuisineModelResult,
    ProcessingStatus as DBProcessingStatus
)
from .core import OperatorCore, ProcessingResult, ProcessingStatus as CoreProcessingStatus
from .pipeline_integration import IntegratedPipeline


class DatabaseIntegratedOperator:
    """
    Operator Layer with database integration for storing results
    and synchronizing with the database after each model completion.
    """
    
    def __init__(self, device: Optional[str] = None, use_bert_cuisine: bool = True):
        """
        Initialize the database-integrated operator.
        
        Args:
            device: Device to run models on
            use_bert_cuisine: Whether to use BERT for cuisine classification
        """
        self.pipeline = IntegratedPipeline(device=device, use_bert_cuisine=use_bert_cuisine)
        self.logger = logging.getLogger(__name__)
    
    async def process_image_with_database(self, image_id: str, image_url: str) -> Dict:
        """
        Process an image through the complete pipeline with database synchronization.
        
        Args:
            image_id: Unique image identifier
            image_url: Public URL of the image to process
            
        Returns:
            Dictionary containing processing results and database status
        """
        db = await get_database()
        db_ops = DatabaseOperations(db)
        
        try:
            self.logger.info(f"Starting database-integrated processing for {image_id}")
            
            # Verify the database record exists
            record = await db_ops.get_analysis_record(image_id)
            if not record:
                raise ValueError(f"No database record found for image_id: {image_id}")
            
            # Add initial processing remark
            initial_remark = ModelRemarkEntry(
                component="operator_layer",
                status="info",
                message=f"Starting integrated processing pipeline for image {image_id}"
            )
            await db_ops.add_model_remark(image_id, initial_remark)
            
            # Process through the integrated pipeline
            pipeline_result = await self.pipeline.process_image_complete(
                image_id=image_id,
                image_url=image_url,
                metadata={"database_integration": True}
            )
            
            # Synchronize results with database
            await self._synchronize_results_with_database(image_id, pipeline_result, db_ops)
            
            # Get final database status
            final_status = await db_ops.get_analysis_status(image_id)
            
            self.logger.info(f"Database-integrated processing completed for {image_id}")
            
            return {
                "image_id": image_id,
                "pipeline_result": pipeline_result,
                "database_status": final_status.dict() if final_status else None,
                "success": pipeline_result["success"],
                "database_synchronized": True
            }
            
        except Exception as e:
            self.logger.error(f"Database-integrated processing failed for {image_id}: {e}")
            
            # Add error remark to database
            try:
                error_remark = ModelRemarkEntry(
                    component="operator_layer",
                    status="error",
                    message=f"Processing pipeline failed: {str(e)}"
                )
                await db_ops.add_model_remark(image_id, error_remark)
                
                # Update database with error status
                await db_ops.update_analysis_record(
                    image_id,
                    UpdateAnalysisRequest(
                        in_progress=False,
                        is_error=True,
                        status=DBProcessingStatus.FAILED
                    )
                )
            except Exception as db_error:
                self.logger.error(f"Failed to update database with error status: {db_error}")
            
            return {
                "image_id": image_id,
                "pipeline_result": None,
                "database_status": None,
                "success": False,
                "error": str(e),
                "database_synchronized": False
            }
    
    async def _synchronize_results_with_database(
        self, 
        image_id: str, 
        pipeline_result: Dict, 
        db_ops: DatabaseOperations
    ):
        """
        Synchronize pipeline results with the database.
        
        Args:
            image_id: Image identifier
            pipeline_result: Results from the integrated pipeline
            db_ops: Database operations instance
        """
        try:
            processing_result = pipeline_result.get("processing_result")
            if not processing_result:
                self.logger.warning(f"No processing result to synchronize for {image_id}")
                return
            
            # Extract model results
            models = processing_result.get("models", {})
            
            # Prepare update request
            update_request = UpdateAnalysisRequest()
            
            # Synchronize Vision Model results
            if "vision" in models:
                vision_data = models["vision"]
                if vision_data.get("status") == "completed" and vision_data.get("data"):
                    vision_result = self._convert_vision_result(vision_data["data"])
                    update_request.vision_result = vision_result
                    
                    # Add success remark
                    remark = ModelRemarkEntry(
                        component="vision_model",
                        status="success",
                        message=f"Vision analysis completed successfully"
                    )
                    await db_ops.add_model_remark(image_id, remark)
                else:
                    # Add error remark
                    error_msg = vision_data.get("error_message", "Vision model failed")
                    remark = ModelRemarkEntry(
                        component="vision_model",
                        status="error",
                        message=f"Vision analysis failed: {error_msg}"
                    )
                    await db_ops.add_model_remark(image_id, remark)
            
            # Synchronize Nutrition Model results
            if "nutrition" in models:
                nutrition_data = models["nutrition"]
                if nutrition_data.get("status") == "completed" and nutrition_data.get("data"):
                    nutrition_result = self._convert_nutrition_result(nutrition_data["data"])
                    update_request.nutrition_result = nutrition_result
                    
                    # Add success remark
                    remark = ModelRemarkEntry(
                        component="nutrition_llm",
                        status="success",
                        message=f"Nutrition analysis completed successfully"
                    )
                    await db_ops.add_model_remark(image_id, remark)
                else:
                    # Add error remark
                    error_msg = nutrition_data.get("error_message", "Nutrition model failed")
                    remark = ModelRemarkEntry(
                        component="nutrition_llm",
                        status="error",
                        message=f"Nutrition analysis failed: {error_msg}"
                    )
                    await db_ops.add_model_remark(image_id, remark)
            
            # Synchronize Cuisine Model results
            if "cuisine" in models:
                cuisine_data = models["cuisine"]
                if cuisine_data.get("status") == "completed" and cuisine_data.get("data"):
                    cuisine_result = self._convert_cuisine_result(cuisine_data["data"])
                    update_request.cuisine_result = cuisine_result
                    
                    # Add success remark
                    remark = ModelRemarkEntry(
                        component="cuisine_classifier",
                        status="success",
                        message=f"Cuisine classification completed successfully"
                    )
                    await db_ops.add_model_remark(image_id, remark)
                else:
                    # Add error remark
                    error_msg = cuisine_data.get("error_message", "Cuisine model failed")
                    remark = ModelRemarkEntry(
                        component="cuisine_classifier",
                        status="error",
                        message=f"Cuisine classification failed: {error_msg}"
                    )
                    await db_ops.add_model_remark(image_id, remark)
            
            # Determine final status
            overall_status = processing_result.get("overall_status", "failed")
            if overall_status == "completed":
                update_request.status = DBProcessingStatus.COMPLETED
                update_request.in_progress = False
                update_request.is_error = False
            elif overall_status == "partial_success":
                update_request.status = DBProcessingStatus.COMPLETED  # Partial success is still completion
                update_request.in_progress = False
                update_request.is_error = True  # Mark as error due to partial failure
            else:
                update_request.status = DBProcessingStatus.FAILED
                update_request.in_progress = False
                update_request.is_error = True
            
            # Update the database record
            updated_record = await db_ops.update_analysis_record(image_id, update_request)
            
            if updated_record:
                # Add final synchronization remark
                final_remark = ModelRemarkEntry(
                    component="operator_layer",
                    status="success",
                    message=f"Database synchronization completed successfully"
                )
                await db_ops.add_model_remark(image_id, final_remark)
                
                self.logger.info(f"Successfully synchronized results for {image_id}")
            else:
                self.logger.error(f"Failed to update database record for {image_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to synchronize results with database for {image_id}: {e}")
            
            # Add error remark
            try:
                error_remark = ModelRemarkEntry(
                    component="operator_layer",
                    status="error",
                    message=f"Database synchronization failed: {str(e)}"
                )
                await db_ops.add_model_remark(image_id, error_remark)
            except Exception:
                pass  # Don't fail if we can't even add the error remark
            
            raise
    
    def _convert_vision_result(self, vision_data: Dict) -> VisionModelResult:
        """
        Convert vision model data to VisionModelResult.
        
        Args:
            vision_data: Raw vision model data
            
        Returns:
            VisionModelResult instance
        """
        return VisionModelResult(
            ingredients=vision_data.get("ingredients", []),
            description=vision_data.get("description", ""),
            confidence=vision_data.get("confidence", 0.0),
            completed=True,
            timestamp=datetime.utcnow()
        )
    
    def _convert_nutrition_result(self, nutrition_data: Dict) -> NutritionModelResult:
        """
        Convert nutrition model data to NutritionModelResult.
        
        Args:
            nutrition_data: Raw nutrition model data
            
        Returns:
            NutritionModelResult instance
        """
        return NutritionModelResult(
            calories=nutrition_data.get("calories", 0.0),
            fat=nutrition_data.get("fat", 0.0),
            carbohydrates=nutrition_data.get("carbohydrates", 0.0),
            protein=nutrition_data.get("protein", 0.0),
            portion_size=nutrition_data.get("portion_size", ""),
            confidence_range=nutrition_data.get("confidence_range", ""),
            completed=True,
            timestamp=datetime.utcnow()
        )
    
    def _convert_cuisine_result(self, cuisine_data: Dict) -> CuisineModelResult:
        """
        Convert cuisine model data to CuisineModelResult.
        
        Args:
            cuisine_data: Raw cuisine model data
            
        Returns:
            CuisineModelResult instance
        """
        from ..database.models import CuisineResult
        
        # Extract cuisine information
        cuisines = []
        
        # Handle different possible formats
        if "cuisines" in cuisine_data:
            # Already in the expected format
            cuisine_list = cuisine_data["cuisines"]
        elif "primary_cuisine" in cuisine_data:
            # Convert from primary/secondary format
            cuisine_list = []
            primary = cuisine_data.get("primary_cuisine", {})
            if primary.get("name"):
                cuisine_list.append({
                    "name": primary["name"],
                    "confidence": primary.get("confidence", 0.0)
                })
            
            # Add secondary cuisines if available
            secondary = cuisine_data.get("secondary_cuisines", [])
            for cuisine in secondary:
                if cuisine.get("name"):
                    cuisine_list.append({
                        "name": cuisine["name"],
                        "confidence": cuisine.get("confidence", 0.0)
                    })
        else:
            cuisine_list = []
        
        # Convert to CuisineResult objects
        for cuisine_info in cuisine_list:
            if isinstance(cuisine_info, dict) and "name" in cuisine_info:
                cuisines.append(CuisineResult(
                    name=cuisine_info["name"],
                    confidence=cuisine_info.get("confidence", 0.0)
                ))
        
        return CuisineModelResult(
            cuisines=cuisines,
            completed=True,
            timestamp=datetime.utcnow()
        )
    
    async def get_processing_status(self, image_id: str) -> Optional[Dict]:
        """
        Get processing status from both pipeline and database.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Combined status information
        """
        try:
            # Get pipeline status
            pipeline_status = self.pipeline.get_processing_status(image_id)
            
            # Get database status
            db = await get_database()
            db_ops = DatabaseOperations(db)
            db_status = await db_ops.get_analysis_status(image_id)
            
            return {
                "image_id": image_id,
                "pipeline_status": pipeline_status,
                "database_status": db_status.dict() if db_status else None,
                "synchronized": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get processing status for {image_id}: {e}")
            return None


# Background task function for API integration
async def trigger_analysis_pipeline(image_id: str, image_url: str) -> Dict:
    """
    Background task function to trigger the analysis pipeline.
    
    This function is designed to be called as a FastAPI background task.
    
    Args:
        image_id: Unique image identifier
        image_url: Public URL of the image to process
        
    Returns:
        Processing result dictionary
    """
    operator = DatabaseIntegratedOperator()
    
    try:
        result = await operator.process_image_with_database(image_id, image_url)
        logging.info(f"Background processing completed for {image_id}: {'SUCCESS' if result['success'] else 'FAILED'}")
        return result
        
    except Exception as e:
        logging.error(f"Background processing failed for {image_id}: {e}")
        
        # Try to update database with error status
        try:
            db = await get_database()
            db_ops = DatabaseOperations(db)
            
            error_remark = ModelRemarkEntry(
                component="background_task",
                status="error",
                message=f"Background processing failed: {str(e)}"
            )
            await db_ops.add_model_remark(image_id, error_remark)
            
            await db_ops.update_analysis_record(
                image_id,
                UpdateAnalysisRequest(
                    in_progress=False,
                    is_error=True,
                    status=DBProcessingStatus.FAILED
                )
            )
        except Exception as db_error:
            logging.error(f"Failed to update database with background task error: {db_error}")
        
        return {
            "image_id": image_id,
            "success": False,
            "error": str(e),
            "database_synchronized": False
        }