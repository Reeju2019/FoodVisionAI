"""
Complete Model Pipeline Integration for FoodVisionAI

Integrates the Operator Core, Logging System, and Status Manager
to provide a unified interface for processing images through all models.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from .core import OperatorCore, ProcessingResult, ModelStage, ProcessingStatus as CoreProcessingStatus
from .logging_system import ModelRemarkLogger, ComponentType, LogLevel, create_logging_session
from .status_manager import StatusManager, ProcessingPhase, ErrorSeverity, ProcessingStatus


class IntegratedPipeline:
    """
    Integrated pipeline that combines all Operator Layer components.
    
    Provides a unified interface for processing images with comprehensive
    logging, status management, and error handling.
    """
    
    def __init__(self, device: Optional[str] = None, use_bert_cuisine: bool = True):
        """
        Initialize the integrated pipeline.
        
        Args:
            device: Device to run models on
            use_bert_cuisine: Whether to use BERT for cuisine classification
        """
        # Initialize components
        self.logger = ModelRemarkLogger()
        self.status_manager = StatusManager(self.logger)
        self.operator_core = OperatorCore(device=device, use_bert_cuisine=use_bert_cuisine)
        
        # Register status callback for logging
        self.status_manager.register_status_callback(self._on_status_change)
        
        logger.info("Integrated Pipeline initialized")
    
    def _on_status_change(self, image_id: str, status: ProcessingStatus):
        """
        Callback for status changes to log important events.
        
        Args:
            image_id: Image identifier
            status: Updated processing status
        """
        self.logger.log_info(
            session_id=image_id,
            component=ComponentType.OPERATOR,
            message=f"Status changed: {status.current_phase.value} (progress: {status.progress_percentage:.1f}%)",
            metadata={
                "phase": status.current_phase.value,
                "progress": status.progress_percentage,
                "in_progress": status.in_progress,
                "is_error": status.is_error
            }
        )
    
    async def process_image_complete(self, image_id: str, image_url: str, 
                                   metadata: Optional[Dict] = None) -> Dict:
        """
        Process an image through the complete pipeline with full integration.
        
        Args:
            image_id: Unique image identifier
            image_url: Public URL of the image to process
            metadata: Optional metadata for the processing session
            
        Returns:
            Dictionary containing complete processing results and status
        """
        # Create logging session
        session_metadata = {
            "image_url": image_url,
            "processing_type": "complete_pipeline",
            **(metadata or {})
        }
        create_logging_session(image_id, session_metadata)
        
        # Initialize processing status
        processing_status = self.status_manager.initialize_processing(image_id, session_metadata)
        
        try:
            logger.info(f"Starting integrated pipeline processing for {image_id}")
            
            # Log pipeline start
            self.logger.log_info(
                session_id=image_id,
                component=ComponentType.OPERATOR,
                message=f"Starting complete pipeline processing for image {image_id}",
                metadata={"image_url": image_url}
            )
            
            # Update to vision processing phase
            self.status_manager.update_phase(image_id, ProcessingPhase.VISION_PROCESSING, 10.0)
            
            # Execute the complete pipeline through operator core
            start_time = time.time()
            processing_result = await self.operator_core.process_image(image_id, image_url)
            total_time = time.time() - start_time
            
            # Process results and update status accordingly
            await self._process_pipeline_results(image_id, processing_result, total_time)
            
            # Get final status
            final_status = self.status_manager.get_status(image_id)
            
            # Create comprehensive response
            response = {
                "image_id": image_id,
                "processing_result": self._format_processing_result(processing_result),
                "status": self.status_manager.get_processing_summary(image_id),
                "session_summary": self.logger.get_session_summary(image_id),
                "success": processing_result.overall_status in [CoreProcessingStatus.COMPLETED, CoreProcessingStatus.PARTIAL_SUCCESS],
                "total_execution_time": total_time
            }
            
            logger.success(f"Integrated pipeline processing completed for {image_id}")
            return response
            
        except Exception as e:
            logger.error(f"Integrated pipeline processing failed for {image_id}: {e}")
            
            # Report error to status manager
            self.status_manager.report_error(
                image_id, 
                f"Pipeline processing failed: {str(e)}",
                ErrorSeverity.CRITICAL,
                {"exception_type": type(e).__name__, "exception_message": str(e)},
                should_continue=False
            )
            
            # Log error
            self.logger.log_error(
                session_id=image_id,
                component=ComponentType.OPERATOR,
                error_message=f"Pipeline processing failed: {str(e)}",
                error_details={"exception_type": type(e).__name__}
            )
            
            # Finalize with failure
            self.status_manager.finalize_processing(image_id, overall_success=False)
            
            return {
                "image_id": image_id,
                "processing_result": None,
                "status": self.status_manager.get_processing_summary(image_id),
                "session_summary": self.logger.get_session_summary(image_id),
                "success": False,
                "error": str(e)
            }
        
        finally:
            # Close logging session
            self.logger.close_session(image_id)
    
    async def _process_pipeline_results(self, image_id: str, result: ProcessingResult, total_time: float):
        """
        Process the results from the operator core and update status accordingly.
        
        Args:
            image_id: Image identifier
            result: Processing result from operator core
            total_time: Total execution time
        """
        # Track model completions
        models_completed = 0
        models_failed = 0
        
        # Process vision model result
        if result.vision_result:
            if result.vision_result.status == CoreProcessingStatus.COMPLETED:
                self.status_manager.mark_model_completed(image_id, "vision", True, result.vision_result.data)
                models_completed += 1
                self.status_manager.update_phase(image_id, ProcessingPhase.NUTRITION_PROCESSING, 33.0)
            else:
                self.status_manager.mark_model_completed(image_id, "vision", False)
                self.status_manager.report_error(
                    image_id,
                    f"Vision model failed: {result.vision_result.error_message}",
                    ErrorSeverity.MEDIUM,
                    should_continue=True
                )
                models_failed += 1
        
        # Process nutrition model result
        if result.nutrition_result:
            if result.nutrition_result.status == CoreProcessingStatus.COMPLETED:
                self.status_manager.mark_model_completed(image_id, "nutrition", True, result.nutrition_result.data)
                models_completed += 1
                self.status_manager.update_phase(image_id, ProcessingPhase.CUISINE_PROCESSING, 66.0)
            else:
                self.status_manager.mark_model_completed(image_id, "nutrition", False)
                self.status_manager.report_error(
                    image_id,
                    f"Nutrition model failed: {result.nutrition_result.error_message}",
                    ErrorSeverity.MEDIUM,
                    should_continue=True
                )
                models_failed += 1
        
        # Process cuisine model result
        if result.cuisine_result:
            if result.cuisine_result.status == CoreProcessingStatus.COMPLETED:
                self.status_manager.mark_model_completed(image_id, "cuisine", True, result.cuisine_result.data)
                models_completed += 1
                self.status_manager.update_phase(image_id, ProcessingPhase.FINALIZATION, 90.0)
            else:
                self.status_manager.mark_model_completed(image_id, "cuisine", False)
                self.status_manager.report_error(
                    image_id,
                    f"Cuisine model failed: {result.cuisine_result.error_message}",
                    ErrorSeverity.MEDIUM,
                    should_continue=True
                )
                models_failed += 1
        
        # Determine overall success
        overall_success = (
            result.overall_status == CoreProcessingStatus.COMPLETED or
            (result.overall_status == CoreProcessingStatus.PARTIAL_SUCCESS and models_completed > 0)
        )
        
        # Finalize processing
        final_result_summary = {
            "models_completed": models_completed,
            "models_failed": models_failed,
            "overall_status": result.overall_status.value,
            "total_execution_time": total_time
        }
        
        self.status_manager.finalize_processing(image_id, overall_success, final_result_summary)
        
        # Log final results
        self.logger.log_info(
            session_id=image_id,
            component=ComponentType.OPERATOR,
            message=f"Pipeline processing completed: {models_completed}/{models_completed + models_failed} models succeeded",
            metadata=final_result_summary
        )
    
    def _format_processing_result(self, result: ProcessingResult) -> Dict:
        """
        Format processing result for API response.
        
        Args:
            result: ProcessingResult from operator core
            
        Returns:
            Formatted result dictionary
        """
        formatted = {
            "image_id": result.image_id,
            "overall_status": result.overall_status.value,
            "success_count": result.success_count,
            "error_count": result.error_count,
            "total_execution_time": result.total_execution_time,
            "models": {}
        }
        
        # Format individual model results
        if result.vision_result:
            formatted["models"]["vision"] = {
                "status": result.vision_result.status.value,
                "data": result.vision_result.data,
                "error_message": result.vision_result.error_message,
                "execution_time": result.vision_result.execution_time,
                "confidence": result.vision_result.confidence
            }
        
        if result.nutrition_result:
            formatted["models"]["nutrition"] = {
                "status": result.nutrition_result.status.value,
                "data": result.nutrition_result.data,
                "error_message": result.nutrition_result.error_message,
                "execution_time": result.nutrition_result.execution_time,
                "confidence": result.nutrition_result.confidence
            }
        
        if result.cuisine_result:
            formatted["models"]["cuisine"] = {
                "status": result.cuisine_result.status.value,
                "data": result.cuisine_result.data,
                "error_message": result.cuisine_result.error_message,
                "execution_time": result.cuisine_result.execution_time,
                "confidence": result.cuisine_result.confidence
            }
        
        return formatted
    
    def get_processing_status(self, image_id: str) -> Optional[Dict]:
        """
        Get current processing status for an image.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Processing status dictionary or None if not found
        """
        return self.status_manager.get_processing_summary(image_id)
    
    def get_session_logs(self, image_id: str) -> Dict:
        """
        Get session logs for an image.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Session logs and summary
        """
        return {
            "session_summary": self.logger.get_session_summary(image_id),
            "remarks": [
                remark.to_dict() 
                for remark in self.logger.get_session_remarks(image_id)
            ]
        }


# Test functions for pipeline validation
async def test_pipeline_with_sample_images():
    """
    Test the complete pipeline with sample food images.
    
    Returns:
        Dictionary with test results
    """
    # Sample food image URLs (these should be publicly accessible)
    test_images = [
        {
            "id": "test_pizza",
            "url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=500",
            "description": "Pizza image for testing"
        },
        {
            "id": "test_salad", 
            "url": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=500",
            "description": "Salad image for testing"
        },
        {
            "id": "test_burger",
            "url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500", 
            "description": "Burger image for testing"
        }
    ]
    
    pipeline = IntegratedPipeline()
    test_results = []
    
    logger.info("Starting pipeline testing with sample images")
    
    for test_image in test_images:
        try:
            logger.info(f"Testing with image: {test_image['description']}")
            
            result = await pipeline.process_image_complete(
                image_id=test_image["id"],
                image_url=test_image["url"],
                metadata={"test_description": test_image["description"]}
            )
            
            test_results.append({
                "image_id": test_image["id"],
                "description": test_image["description"],
                "success": result["success"],
                "execution_time": result.get("total_execution_time", 0),
                "models_completed": result["processing_result"]["success_count"] if result["processing_result"] else 0,
                "errors": result["processing_result"]["error_count"] if result["processing_result"] else 1
            })
            
            logger.info(f"Test completed for {test_image['id']}: {'SUCCESS' if result['success'] else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Test failed for {test_image['id']}: {e}")
            test_results.append({
                "image_id": test_image["id"],
                "description": test_image["description"],
                "success": False,
                "error": str(e)
            })
    
    # Calculate summary statistics
    successful_tests = sum(1 for result in test_results if result["success"])
    total_tests = len(test_results)
    
    summary = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
        "test_results": test_results
    }
    
    logger.info(f"Pipeline testing completed: {successful_tests}/{total_tests} tests passed")
    
    return summary


async def test_error_scenarios():
    """
    Test pipeline behavior with error scenarios.
    
    Returns:
        Dictionary with error test results
    """
    pipeline = IntegratedPipeline()
    error_tests = []
    
    logger.info("Starting error scenario testing")
    
    # Test with invalid URL
    try:
        result = await pipeline.process_image_complete(
            image_id="test_invalid_url",
            image_url="https://invalid-url-that-does-not-exist.com/image.jpg",
            metadata={"test_type": "invalid_url"}
        )
        
        error_tests.append({
            "test_name": "invalid_url",
            "expected_failure": True,
            "actual_success": result["success"],
            "handled_gracefully": not result["success"]  # Should fail gracefully
        })
        
    except Exception as e:
        error_tests.append({
            "test_name": "invalid_url",
            "expected_failure": True,
            "actual_success": False,
            "handled_gracefully": True,
            "error": str(e)
        })
    
    # Test with non-image URL
    try:
        result = await pipeline.process_image_complete(
            image_id="test_non_image",
            image_url="https://www.google.com",
            metadata={"test_type": "non_image_url"}
        )
        
        error_tests.append({
            "test_name": "non_image_url",
            "expected_failure": True,
            "actual_success": result["success"],
            "handled_gracefully": not result["success"]
        })
        
    except Exception as e:
        error_tests.append({
            "test_name": "non_image_url",
            "expected_failure": True,
            "actual_success": False,
            "handled_gracefully": True,
            "error": str(e)
        })
    
    summary = {
        "total_error_tests": len(error_tests),
        "gracefully_handled": sum(1 for test in error_tests if test["handled_gracefully"]),
        "error_tests": error_tests
    }
    
    logger.info(f"Error scenario testing completed: {summary['gracefully_handled']}/{summary['total_error_tests']} handled gracefully")
    
    return summary


# Convenience function for quick testing
async def run_complete_pipeline_test():
    """
    Run complete pipeline test including normal and error scenarios.
    
    Returns:
        Comprehensive test results
    """
    logger.info("Starting complete pipeline testing")
    
    # Run normal tests
    normal_results = await test_pipeline_with_sample_images()
    
    # Run error tests
    error_results = await test_error_scenarios()
    
    # Combine results
    complete_results = {
        "test_timestamp": datetime.now().isoformat(),
        "normal_scenarios": normal_results,
        "error_scenarios": error_results,
        "overall_summary": {
            "total_tests": normal_results["total_tests"] + error_results["total_error_tests"],
            "normal_success_rate": normal_results["success_rate"],
            "error_handling_rate": error_results["gracefully_handled"] / error_results["total_error_tests"] if error_results["total_error_tests"] > 0 else 1.0
        }
    }
    
    logger.info("Complete pipeline testing finished")
    
    return complete_results