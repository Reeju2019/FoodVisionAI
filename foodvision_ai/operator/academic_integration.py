"""
Academic Pipeline Integration for Operator Layer

Integrates the 3-stage academic pipeline with the existing operator system
for seamless database integration and status tracking.
"""

import asyncio
from typing import Dict, Any
from loguru import logger

from ..models.academic_pipeline import AcademicFoodPipeline
from ..database import DatabaseOperations
from ..database.models import ModelRemarkEntry


class AcademicPipelineOperator:
    """
    Operator for managing academic 3-stage pipeline execution
    with database integration and status tracking
    """
    
    def __init__(self):
        self.pipeline = AcademicFoodPipeline()
        logger.info("Academic Pipeline Operator initialized")
    
    async def process_image_academic(self, image_id: str, image_url: str, db_ops: DatabaseOperations) -> Dict[str, Any]:
        """
        Process image through academic 3-stage pipeline with full tracking
        
        Args:
            image_id: Unique image identifier
            image_url: URL or path to image
            db_ops: Database operations instance
            
        Returns:
            Complete analysis results
        """
        logger.info(f"🎓 Starting academic pipeline for {image_id}")
        
        try:
            # Update status to processing
            await db_ops.update_analysis_status(image_id, "processing")
            
            # Add initial remark
            await db_ops.add_model_remark(image_id, ModelRemarkEntry(
                component="academic_pipeline",
                status="info",
                message="Starting 3-stage academic analysis pipeline"
            ))
            
            # Execute academic pipeline
            results = self.pipeline.analyze_food_image(image_url)
            
            if results["analysis_status"] == "success":
                # Convert academic results to database format
                db_results = self._convert_academic_results(results)
                
                # Update database with results
                await db_ops.update_analysis_results(image_id, db_results)
                
                # Add success remarks for each stage
                await self._add_stage_remarks(image_id, results, db_ops)
                
                # Mark as completed
                await db_ops.update_analysis_status(image_id, "completed")
                
                logger.success(f"🎓 Academic pipeline completed for {image_id}")
                return results
            else:
                # Handle failure
                await db_ops.add_model_remark(image_id, ModelRemarkEntry(
                    component="academic_pipeline",
                    status="error",
                    message=f"Pipeline failed: {results.get('error_message', 'Unknown error')}"
                ))
                
                await db_ops.update_analysis_status(image_id, "failed")
                return results
                
        except Exception as e:
            logger.error(f"Academic pipeline error for {image_id}: {e}")
            
            await db_ops.add_model_remark(image_id, ModelRemarkEntry(
                component="academic_pipeline",
                status="error",
                message=f"Pipeline exception: {str(e)}"
            ))
            
            await db_ops.update_analysis_status(image_id, "failed")
            
            return {
                "analysis_status": "failed",
                "error_message": str(e),
                "stage1_ingredients": {"ingredients": []},
                "stage2_dish_analysis": {"predicted_dish": "unknown", "description": "Analysis failed"},
                "stage3_nutrition": {"calories": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0}
            }
    
    def _convert_academic_results(self, academic_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert academic pipeline results to database-compatible format
        
        Args:
            academic_results: Results from academic pipeline
            
        Returns:
            Database-compatible results format
        """
        stage1 = academic_results.get("stage1_ingredients", {})
        stage2 = academic_results.get("stage2_dish_analysis", {})
        stage3 = academic_results.get("stage3_nutrition", {})
        
        # Convert to existing database format for compatibility
        db_results = {
            "vision": {
                "ingredients": [ing["name"] for ing in stage1.get("ingredients", [])],
                "description": stage2.get("description", ""),
                "confidence": stage1.get("model_confidence", 0.0),
                "completed": True,
                "timestamp": None,  # Will be set by database
                "academic_stage1": stage1,  # Keep original academic data
                "academic_stage2": stage2
            },
            "nutrition": {
                "calories": stage3.get("calories", 0),
                "protein": stage3.get("protein_g", 0),
                "fat": stage3.get("fat_g", 0),
                "carbohydrates": stage3.get("carbs_g", 0),
                "portion_size": stage3.get("portion_size", "1 serving"),
                "confidence_range": "±10%",  # Academic database confidence
                "completed": True,
                "timestamp": None,
                "academic_stage3": stage3  # Keep original academic data
            },
            "cuisine": {
                "cuisines": [{
                    "name": stage2.get("cuisine", "unknown").title(),
                    "confidence": stage2.get("confidence", 0.0)
                }],
                "completed": True,
                "timestamp": None,
                "academic_reasoning": stage2.get("reasoning_method", "unknown")
            }
        }
        
        return db_results
    
    async def _add_stage_remarks(self, image_id: str, results: Dict[str, Any], db_ops: DatabaseOperations):
        """Add detailed remarks for each academic pipeline stage"""
        
        # Stage 1 remark
        stage1 = results.get("stage1_ingredients", {})
        ingredients_count = len(stage1.get("ingredients", []))
        confidence = stage1.get("model_confidence", 0.0)
        
        await db_ops.add_model_remark(image_id, ModelRemarkEntry(
            component="stage1_cnn",
            status="success",
            message=f"CNN detected {ingredients_count} ingredients with {confidence:.1%} confidence",
            data={
                "ingredients_detected": ingredients_count,
                "model_confidence": confidence,
                "detection_method": stage1.get("detection_method", "unknown")
            }
        ))
        
        # Stage 2 remark
        stage2 = results.get("stage2_dish_analysis", {})
        predicted_dish = stage2.get("predicted_dish", "unknown")
        reasoning_method = stage2.get("reasoning_method", "unknown")
        
        await db_ops.add_model_remark(image_id, ModelRemarkEntry(
            component="stage2_genai",
            status="success",
            message=f"GenAI identified dish as '{predicted_dish}' using {reasoning_method}",
            data={
                "predicted_dish": predicted_dish,
                "cuisine": stage2.get("cuisine", "unknown"),
                "reasoning_method": reasoning_method,
                "confidence": stage2.get("confidence", 0.0)
            }
        ))
        
        # Stage 3 remark
        stage3 = results.get("stage3_nutrition", {})
        calories = stage3.get("calories", 0)
        lookup_method = stage3.get("lookup_method", "unknown")
        
        await db_ops.add_model_remark(image_id, ModelRemarkEntry(
            component="stage3_nutrition",
            status="success",
            message=f"Nutrition lookup found {calories} calories using {lookup_method}",
            data={
                "calories": calories,
                "protein_g": stage3.get("protein_g", 0),
                "fat_g": stage3.get("fat_g", 0),
                "carbs_g": stage3.get("carbs_g", 0),
                "lookup_method": lookup_method
            }
        ))
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get academic pipeline information"""
        return self.pipeline.get_pipeline_info()


# Global academic operator instance
academic_operator = AcademicPipelineOperator()


async def trigger_academic_analysis_pipeline(image_id: str, image_url: str):
    """
    Trigger academic analysis pipeline (background task)
    
    Args:
        image_id: Unique image identifier
        image_url: URL or path to image
    """
    try:
        from ..database import db_connection
        from ..database import DatabaseOperations
        
        # Get database connection
        if db_connection.database is None:
            logger.warning("Database not connected, skipping academic analysis")
            return
        
        db_ops = DatabaseOperations(db_connection.database)
        
        # Execute academic pipeline
        results = await academic_operator.process_image_academic(image_id, image_url, db_ops)
        
        logger.info(f"Academic analysis completed for {image_id}: {results['analysis_status']}")
        
    except Exception as e:
        logger.error(f"Academic analysis pipeline failed for {image_id}: {e}")
        
        # Try to update status to failed
        try:
            from ..database import db_connection, DatabaseOperations
            if db_connection.database is not None:
                db_ops = DatabaseOperations(db_connection.database)
                await db_ops.update_analysis_status(image_id, "failed")
                await db_ops.add_model_remark(image_id, ModelRemarkEntry(
                    component="academic_pipeline",
                    status="error",
                    message=f"Pipeline exception: {str(e)}"
                ))
        except:
            pass  # If database update fails, just log the original error