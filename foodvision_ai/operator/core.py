"""
Operator Layer Core Functionality for FoodVisionAI

Implements async model execution orchestration, comprehensive error handling,
and model pipeline coordination for processing images through all three stages.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
from loguru import logger

from ..models.vision_model import VisionModel
from ..models.nutrition_llm import NutritionLLM
from ..models.cuisine_classifier import CuisineClassifier
from ..services.model_backend_switcher import ModelBackendSwitcher, SwitchingStrategy


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class ModelStage(Enum):
    """Model processing stages."""
    VISION = "vision"
    NUTRITION = "nutrition"
    CUISINE = "cuisine"


@dataclass
class ModelResult:
    """Result from a single model execution."""
    stage: ModelStage
    status: ProcessingStatus
    data: Optional[Dict] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ProcessingResult:
    """Complete processing result from all models."""
    image_id: str
    overall_status: ProcessingStatus
    vision_result: Optional[ModelResult] = None
    nutrition_result: Optional[ModelResult] = None
    cuisine_result: Optional[ModelResult] = None
    total_execution_time: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    model_remarks: List[Dict] = None
    
    def __post_init__(self):
        if self.model_remarks is None:
            self.model_remarks = []


class OperatorCore:
    """
    Core Operator Layer for orchestrating model execution and handling errors.
    
    Manages async execution of Vision Model, Nutrition LLM, and Cuisine Classifier
    with comprehensive error handling and logging.
    """
    
    def __init__(self, device: Optional[str] = None, use_bert_cuisine: bool = True, 
                 use_colab: bool = False, switching_strategy: SwitchingStrategy = SwitchingStrategy.LOCAL_ONLY):
        """
        Initialize the Operator Core.
        
        Args:
            device: Device to run models on ('cpu', 'cuda', or None for auto)
            use_bert_cuisine: Whether to use BERT for cuisine classification
            use_colab: Whether to enable Colab backend integration
            switching_strategy: Strategy for backend switching when Colab is enabled
        """
        self.device = device
        self.use_bert_cuisine = use_bert_cuisine
        self.use_colab = use_colab
        self.switching_strategy = switching_strategy
        
        # Initialize models (legacy for direct local access)
        self.vision_model = None
        self.nutrition_llm = None
        self.cuisine_classifier = None
        
        # Backend switcher for Colab integration
        self.backend_switcher: Optional[ModelBackendSwitcher] = None
        
        # Processing state
        self.active_processing = {}  # Track active processing sessions
        
        logger.info(f"Operator Core initialized (Colab: {use_colab}, Strategy: {switching_strategy.value if use_colab else 'N/A'})")
    
    def register_colab_endpoint(self, model_type: str, ngrok_url: str, endpoint_path: str = "/predict") -> bool:
        """
        Register a Colab endpoint for a specific model type.
        
        Args:
            model_type: Type of model ("vision", "nutrition", "cuisine")
            ngrok_url: Base ngrok URL from Colab
            endpoint_path: API endpoint path
            
        Returns:
            True if registration successful, False otherwise
        """
        if not self.use_colab:
            logger.error("Colab integration not enabled")
            return False
        
        if not self.backend_switcher:
            logger.error("Backend switcher not initialized")
            return False
        
        return self.backend_switcher.register_colab_endpoint(model_type, ngrok_url, endpoint_path)
    
    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get backend status information.
        
        Returns:
            Dictionary with backend status, or None if Colab not enabled
        """
        if self.use_colab and self.backend_switcher:
            return self.backend_switcher.get_backend_status()
        else:
            return {
                "colab_enabled": False,
                "local_models_active": True,
                "switching_strategy": "local_only"
            }
    
    async def set_switching_strategy(self, strategy: SwitchingStrategy):
        """
        Update the backend switching strategy.
        
        Args:
            strategy: New switching strategy
        """
        if self.use_colab and self.backend_switcher:
            self.backend_switcher.set_switching_strategy(strategy)
            self.switching_strategy = strategy
            logger.info(f"Switching strategy updated to: {strategy.value}")
        else:
            logger.warning("Cannot change switching strategy - Colab integration not enabled")
    
    async def _initialize_models(self):
        """Initialize all models lazily when first needed."""
        try:
            if self.use_colab:
                # Initialize backend switcher for Colab integration
                if self.backend_switcher is None:
                    logger.info("Initializing Backend Switcher with Colab support...")
                    self.backend_switcher = ModelBackendSwitcher(
                        switching_strategy=self.switching_strategy,
                        device=self.device
                    )
                    await self.backend_switcher.initialize()
                    logger.success("Backend Switcher initialized successfully")
            else:
                # Initialize local models directly (legacy mode)
                if self.vision_model is None:
                    logger.info("Initializing Vision Model...")
                    self.vision_model = VisionModel(device=self.device)
                
                if self.nutrition_llm is None:
                    logger.info("Initializing Nutrition LLM...")
                    self.nutrition_llm = NutritionLLM(device=self.device)
                
                if self.cuisine_classifier is None:
                    logger.info("Initializing Cuisine Classifier...")
                    self.cuisine_classifier = CuisineClassifier(
                        use_bert=self.use_bert_cuisine, 
                        device=self.device
                    )
                
                logger.success("All local models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _create_model_remark(self, stage: str, status: str, message: str, 
                           data: Optional[Dict] = None, error: Optional[str] = None) -> Dict:
        """
        Create a standardized model remark entry.
        
        Args:
            stage: Model stage (vision, nutrition, cuisine)
            status: Status (success, error, warning, info)
            message: Human-readable message
            data: Optional data payload
            error: Optional error details
            
        Returns:
            Dictionary containing the model remark
        """
        remark = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "status": status,
            "message": message
        }
        
        if data is not None:
            remark["data"] = data
        
        if error is not None:
            remark["error"] = error
        
        return remark
    
    async def _execute_vision_model(self, image_url: str, model_remarks: List[Dict]) -> ModelResult:
        """
        Execute vision model with comprehensive error handling.
        
        Args:
            image_url: URL of the image to analyze
            model_remarks: List to append execution logs to
            
        Returns:
            ModelResult containing vision analysis results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Starting vision model execution for image: {image_url}")
            
            # Add start remark
            model_remarks.append(self._create_model_remark(
                "vision", "info", f"Starting vision analysis for image: {image_url}"
            ))
            
            # Execute vision model using appropriate backend
            if self.use_colab and self.backend_switcher:
                # Use backend switcher for Colab integration
                response = await self.backend_switcher.predict_vision(image_url)
                
                if response.success:
                    vision_data = response.data
                    backend_info = f" via {response.backend_used.value} backend"
                    if response.fallback_used:
                        backend_info += " (fallback)"
                else:
                    raise Exception(response.error_message)
            else:
                # Use local model directly
                vision_data = self.vision_model.analyze_image(image_url)
                backend_info = " via local backend"
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Check if analysis was successful
            if vision_data.get('analysis_status') == 'success':
                confidence = vision_data.get('confidence', 0.0)
                
                model_remarks.append(self._create_model_remark(
                    "vision", "success", 
                    f"Vision analysis completed successfully in {execution_time:.2f}s{backend_info}",
                    data={"confidence": confidence, "ingredients_count": len(vision_data.get('ingredients', []))}
                ))
                
                return ModelResult(
                    stage=ModelStage.VISION,
                    status=ProcessingStatus.COMPLETED,
                    data=vision_data,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat(),
                    confidence=confidence
                )
            else:
                error_msg = vision_data.get('error_message', 'Unknown vision model error')
                
                model_remarks.append(self._create_model_remark(
                    "vision", "error", 
                    f"Vision analysis failed: {error_msg}",
                    error=error_msg
                ))
                
                return ModelResult(
                    stage=ModelStage.VISION,
                    status=ProcessingStatus.FAILED,
                    error_message=error_msg,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Vision model execution failed: {str(e)}"
            
            logger.error(error_msg)
            logger.error(f"Vision model traceback: {traceback.format_exc()}")
            
            model_remarks.append(self._create_model_remark(
                "vision", "error", error_msg, error=traceback.format_exc()
            ))
            
            return ModelResult(
                stage=ModelStage.VISION,
                status=ProcessingStatus.FAILED,
                error_message=error_msg,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def _execute_nutrition_model(self, vision_data: Dict, model_remarks: List[Dict]) -> ModelResult:
        """
        Execute nutrition model with comprehensive error handling.
        
        Args:
            vision_data: Results from vision model
            model_remarks: List to append execution logs to
            
        Returns:
            ModelResult containing nutrition analysis results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info("Starting nutrition model execution")
            
            # Add start remark
            model_remarks.append(self._create_model_remark(
                "nutrition", "info", "Starting nutrition analysis from vision results"
            ))
            
            # Execute nutrition model using appropriate backend
            if self.use_colab and self.backend_switcher:
                # Use backend switcher for Colab integration
                ingredients = vision_data.get('ingredients', [])
                description = vision_data.get('description', '')
                response = await self.backend_switcher.predict_nutrition(ingredients, description)
                
                if response.success:
                    nutrition_data = response.data
                    backend_info = f" via {response.backend_used.value} backend"
                    if response.fallback_used:
                        backend_info += " (fallback)"
                else:
                    raise Exception(response.error_message)
            else:
                # Use local model directly
                nutrition_data = self.nutrition_llm.analyze_nutrition(vision_data)
                backend_info = " via local backend"
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Check if analysis was successful
            if nutrition_data.get('analysis_status') == 'success':
                confidence = nutrition_data.get('confidence', 0.0)
                
                model_remarks.append(self._create_model_remark(
                    "nutrition", "success", 
                    f"Nutrition analysis completed successfully in {execution_time:.2f}s{backend_info}",
                    data={
                        "confidence": confidence, 
                        "calories": nutrition_data.get('calories', 0),
                        "confidence_level": nutrition_data.get('confidence_level', 'unknown')
                    }
                ))
                
                return ModelResult(
                    stage=ModelStage.NUTRITION,
                    status=ProcessingStatus.COMPLETED,
                    data=nutrition_data,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat(),
                    confidence=confidence
                )
            else:
                error_msg = nutrition_data.get('error_message', 'Unknown nutrition model error')
                
                model_remarks.append(self._create_model_remark(
                    "nutrition", "error", 
                    f"Nutrition analysis failed: {error_msg}",
                    error=error_msg
                ))
                
                return ModelResult(
                    stage=ModelStage.NUTRITION,
                    status=ProcessingStatus.FAILED,
                    error_message=error_msg,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Nutrition model execution failed: {str(e)}"
            
            logger.error(error_msg)
            logger.error(f"Nutrition model traceback: {traceback.format_exc()}")
            
            model_remarks.append(self._create_model_remark(
                "nutrition", "error", error_msg, error=traceback.format_exc()
            ))
            
            return ModelResult(
                stage=ModelStage.NUTRITION,
                status=ProcessingStatus.FAILED,
                error_message=error_msg,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def _execute_cuisine_model(self, vision_data: Dict, model_remarks: List[Dict]) -> ModelResult:
        """
        Execute cuisine classifier with comprehensive error handling.
        
        Args:
            vision_data: Results from vision model
            model_remarks: List to append execution logs to
            
        Returns:
            ModelResult containing cuisine analysis results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info("Starting cuisine model execution")
            
            # Add start remark
            model_remarks.append(self._create_model_remark(
                "cuisine", "info", "Starting cuisine classification from vision results"
            ))
            
            # Execute cuisine classifier using appropriate backend
            if self.use_colab and self.backend_switcher:
                # Use backend switcher for Colab integration
                ingredients = vision_data.get('ingredients', [])
                description = vision_data.get('description', '')
                response = await self.backend_switcher.predict_cuisine(ingredients, description)
                
                if response.success:
                    cuisine_data = response.data
                    backend_info = f" via {response.backend_used.value} backend"
                    if response.fallback_used:
                        backend_info += " (fallback)"
                else:
                    raise Exception(response.error_message)
            else:
                # Use local model directly
                cuisine_data = self.cuisine_classifier.analyze_cuisine(vision_data)
                backend_info = " via local backend"
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Check if analysis was successful
            if cuisine_data.get('analysis_status') == 'success':
                confidence = cuisine_data.get('total_confidence', 0.0)
                primary_cuisine = cuisine_data.get('primary_cuisine', {}).get('name', 'Unknown')
                
                model_remarks.append(self._create_model_remark(
                    "cuisine", "success", 
                    f"Cuisine classification completed successfully in {execution_time:.2f}s{backend_info}",
                    data={
                        "confidence": confidence, 
                        "primary_cuisine": primary_cuisine,
                        "multiple_cuisines": cuisine_data.get('multiple_cuisines_detected', False)
                    }
                ))
                
                return ModelResult(
                    stage=ModelStage.CUISINE,
                    status=ProcessingStatus.COMPLETED,
                    data=cuisine_data,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat(),
                    confidence=confidence
                )
            else:
                error_msg = cuisine_data.get('error_message', 'Unknown cuisine model error')
                
                model_remarks.append(self._create_model_remark(
                    "cuisine", "error", 
                    f"Cuisine classification failed: {error_msg}",
                    error=error_msg
                ))
                
                return ModelResult(
                    stage=ModelStage.CUISINE,
                    status=ProcessingStatus.FAILED,
                    error_message=error_msg,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Cuisine model execution failed: {str(e)}"
            
            logger.error(error_msg)
            logger.error(f"Cuisine model traceback: {traceback.format_exc()}")
            
            model_remarks.append(self._create_model_remark(
                "cuisine", "error", error_msg, error=traceback.format_exc()
            ))
            
            return ModelResult(
                stage=ModelStage.CUISINE,
                status=ProcessingStatus.FAILED,
                error_message=error_msg,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def process_image(self, image_id: str, image_url: str) -> ProcessingResult:
        """
        Process an image through the complete model pipeline.
        
        Args:
            image_id: Unique identifier for the image
            image_url: Public URL of the image to process
            
        Returns:
            ProcessingResult containing results from all models
        """
        overall_start_time = asyncio.get_event_loop().time()
        model_remarks = []
        
        try:
            logger.info(f"Starting complete image processing for {image_id}")
            
            # Initialize models if not already done
            await self._initialize_models()
            
            # Track processing session
            self.active_processing[image_id] = {
                "start_time": overall_start_time,
                "status": ProcessingStatus.IN_PROGRESS
            }
            
            # Add initial processing remark
            model_remarks.append(self._create_model_remark(
                "operator", "info", f"Starting complete processing pipeline for image {image_id}"
            ))
            
            # Execute Vision Model
            vision_result = await self._execute_vision_model(image_url, model_remarks)
            
            # Initialize results tracking
            success_count = 0
            error_count = 0
            
            if vision_result.status == ProcessingStatus.COMPLETED:
                success_count += 1
                vision_data = vision_result.data
            else:
                error_count += 1
                # Create fallback vision data for downstream models
                vision_data = {
                    'ingredients': [],
                    'description': 'Vision analysis failed',
                    'analysis_status': 'error'
                }
            
            # Execute Nutrition and Cuisine models in parallel (they both depend on vision)
            nutrition_task = self._execute_nutrition_model(vision_data, model_remarks)
            cuisine_task = self._execute_cuisine_model(vision_data, model_remarks)
            
            # Wait for both models to complete
            nutrition_result, cuisine_result = await asyncio.gather(
                nutrition_task, cuisine_task, return_exceptions=True
            )
            
            # Handle potential exceptions from gather
            if isinstance(nutrition_result, Exception):
                logger.error(f"Nutrition model task failed: {nutrition_result}")
                model_remarks.append(self._create_model_remark(
                    "nutrition", "error", f"Nutrition task failed: {str(nutrition_result)}"
                ))
                nutrition_result = ModelResult(
                    stage=ModelStage.NUTRITION,
                    status=ProcessingStatus.FAILED,
                    error_message=str(nutrition_result),
                    timestamp=datetime.now().isoformat()
                )
            
            if isinstance(cuisine_result, Exception):
                logger.error(f"Cuisine model task failed: {cuisine_result}")
                model_remarks.append(self._create_model_remark(
                    "cuisine", "error", f"Cuisine task failed: {str(cuisine_result)}"
                ))
                cuisine_result = ModelResult(
                    stage=ModelStage.CUISINE,
                    status=ProcessingStatus.FAILED,
                    error_message=str(cuisine_result),
                    timestamp=datetime.now().isoformat()
                )
            
            # Update success/error counts
            if nutrition_result.status == ProcessingStatus.COMPLETED:
                success_count += 1
            else:
                error_count += 1
            
            if cuisine_result.status == ProcessingStatus.COMPLETED:
                success_count += 1
            else:
                error_count += 1
            
            # Calculate total execution time
            total_execution_time = asyncio.get_event_loop().time() - overall_start_time
            
            # Determine overall status
            if success_count == 3:
                overall_status = ProcessingStatus.COMPLETED
                status_message = "All models completed successfully"
            elif success_count > 0:
                overall_status = ProcessingStatus.PARTIAL_SUCCESS
                status_message = f"{success_count}/3 models completed successfully"
            else:
                overall_status = ProcessingStatus.FAILED
                status_message = "All models failed"
            
            # Add final processing remark
            model_remarks.append(self._create_model_remark(
                "operator", "success" if overall_status == ProcessingStatus.COMPLETED else "warning",
                f"Processing completed: {status_message} in {total_execution_time:.2f}s",
                data={
                    "success_count": success_count,
                    "error_count": error_count,
                    "total_time": total_execution_time
                }
            ))
            
            # Create final result
            result = ProcessingResult(
                image_id=image_id,
                overall_status=overall_status,
                vision_result=vision_result,
                nutrition_result=nutrition_result,
                cuisine_result=cuisine_result,
                total_execution_time=total_execution_time,
                error_count=error_count,
                success_count=success_count,
                model_remarks=model_remarks
            )
            
            # Update processing session
            self.active_processing[image_id] = {
                "start_time": overall_start_time,
                "status": overall_status,
                "result": result
            }
            
            logger.success(f"Image processing completed for {image_id}: {status_message}")
            
            return result
            
        except Exception as e:
            total_execution_time = asyncio.get_event_loop().time() - overall_start_time
            error_msg = f"Complete processing pipeline failed: {str(e)}"
            
            logger.error(error_msg)
            logger.error(f"Pipeline traceback: {traceback.format_exc()}")
            
            model_remarks.append(self._create_model_remark(
                "operator", "error", error_msg, error=traceback.format_exc()
            ))
            
            # Create failed result
            result = ProcessingResult(
                image_id=image_id,
                overall_status=ProcessingStatus.FAILED,
                total_execution_time=total_execution_time,
                error_count=3,  # Assume all models failed
                success_count=0,
                model_remarks=model_remarks
            )
            
            # Update processing session
            self.active_processing[image_id] = {
                "start_time": overall_start_time,
                "status": ProcessingStatus.FAILED,
                "result": result
            }
            
            return result
        
        finally:
            # Clean up processing session after some time (optional)
            # In a real implementation, you might want to clean this up periodically
            pass
    
    async def cleanup(self):
        """Cleanup resources including backend switcher."""
        if self.backend_switcher:
            await self.backend_switcher.cleanup()
            logger.debug("Backend switcher cleaned up")
    
    def get_processing_status(self, image_id: str) -> Optional[Dict]:
        """
        Get current processing status for an image.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Dictionary with processing status or None if not found
        """
        return self.active_processing.get(image_id)
    
    def get_active_processing_count(self) -> int:
        """
        Get count of currently active processing sessions.
        
        Returns:
            Number of active processing sessions
        """
        active_count = sum(
            1 for session in self.active_processing.values()
            if session["status"] == ProcessingStatus.IN_PROGRESS
        )
        return active_count
    
    def cleanup_completed_sessions(self, max_age_hours: int = 24):
        """
        Clean up old completed processing sessions.
        
        Args:
            max_age_hours: Maximum age in hours for keeping completed sessions
        """
        current_time = asyncio.get_event_loop().time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for image_id, session in self.active_processing.items():
            if (current_time - session["start_time"]) > max_age_seconds:
                if session["status"] != ProcessingStatus.IN_PROGRESS:
                    to_remove.append(image_id)
        
        for image_id in to_remove:
            del self.active_processing[image_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old processing sessions")


# Convenience functions for quick processing
async def process_image_complete(image_id: str, image_url: str, 
                               device: Optional[str] = None, use_colab: bool = False,
                               switching_strategy: SwitchingStrategy = SwitchingStrategy.LOCAL_ONLY) -> ProcessingResult:
    """
    Convenience function to process an image through the complete pipeline.
    
    Args:
        image_id: Unique identifier for the image
        image_url: Public URL of the image to process
        device: Device to run models on
        use_colab: Whether to enable Colab backend integration
        switching_strategy: Strategy for backend switching
        
    Returns:
        ProcessingResult containing results from all models
    """
    operator = OperatorCore(device=device, use_colab=use_colab, switching_strategy=switching_strategy)
    try:
        return await operator.process_image(image_id, image_url)
    finally:
        await operator.cleanup()


async def process_image_with_colab(image_id: str, image_url: str, 
                                 colab_endpoints: Dict[str, str],
                                 device: Optional[str] = None,
                                 switching_strategy: SwitchingStrategy = SwitchingStrategy.COLAB_FIRST) -> ProcessingResult:
    """
    Convenience function to process an image using Colab endpoints with fallback.
    
    Args:
        image_id: Unique identifier for the image
        image_url: Public URL of the image to process
        colab_endpoints: Dictionary mapping model types to ngrok URLs
        device: Device to run local models on (for fallback)
        switching_strategy: Strategy for backend switching
        
    Returns:
        ProcessingResult containing results from all models
    """
    operator = OperatorCore(device=device, use_colab=True, switching_strategy=switching_strategy)
    
    try:
        # Register Colab endpoints
        for model_type, ngrok_url in colab_endpoints.items():
            success = operator.register_colab_endpoint(model_type, ngrok_url)
            if success:
                logger.info(f"Registered {model_type} Colab endpoint: {ngrok_url}")
            else:
                logger.warning(f"Failed to register {model_type} Colab endpoint: {ngrok_url}")
        
        # Process image
        return await operator.process_image(image_id, image_url)
        
    finally:
        await operator.cleanup()