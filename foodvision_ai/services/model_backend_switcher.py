"""
Model Backend Switcher for FoodVisionAI

Implements automatic fallback from Colab to local models with session expiration detection
and recovery. Ensures API interface consistency across backends.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .colab_client import ColabClient, ColabModelType, ColabConnectionStatus, ColabResponse
from ..models.vision_model import VisionModel
from ..models.nutrition_llm import NutritionLLM
from ..models.cuisine_classifier import CuisineClassifier


class BackendType(Enum):
    """Types of model backends."""
    COLAB = "colab"
    LOCAL = "local"


class SwitchingStrategy(Enum):
    """Strategies for backend switching."""
    COLAB_FIRST = "colab_first"  # Try Colab first, fallback to local
    LOCAL_FIRST = "local_first"  # Try local first, fallback to Colab
    COLAB_ONLY = "colab_only"    # Only use Colab (fail if unavailable)
    LOCAL_ONLY = "local_only"    # Only use local models


@dataclass
class BackendStatus:
    """Status information for a model backend."""
    backend_type: BackendType
    is_available: bool
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    average_response_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0


@dataclass
class ModelRequest:
    """Standardized model request structure."""
    model_type: str  # "vision", "nutrition", "cuisine"
    input_data: Dict[str, Any]
    timeout_seconds: int = 30
    retry_count: int = 0


@dataclass
class ModelResponse:
    """Standardized model response structure."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    backend_used: Optional[BackendType] = None
    response_time: Optional[float] = None
    fallback_used: bool = False
    retry_count: int = 0


class ModelBackendSwitcher:
    """
    Manages automatic switching between Colab and local model backends.
    
    Provides session expiration detection, automatic fallback, and consistent
    API interface regardless of backend used.
    """
    
    def __init__(self, 
                 switching_strategy: SwitchingStrategy = SwitchingStrategy.COLAB_FIRST,
                 session_timeout: int = 3600,  # 1 hour
                 failure_threshold: int = 3,
                 health_check_interval: int = 300,  # 5 minutes
                 device: Optional[str] = None):
        """
        Initialize the Model Backend Switcher.
        
        Args:
            switching_strategy: Strategy for choosing backends
            session_timeout: Colab session timeout in seconds
            failure_threshold: Number of consecutive failures before switching
            health_check_interval: Interval between health checks in seconds
            device: Device for local models ('cpu', 'cuda', or None for auto)
        """
        self.switching_strategy = switching_strategy
        self.session_timeout = session_timeout
        self.failure_threshold = failure_threshold
        self.health_check_interval = health_check_interval
        self.device = device
        
        # Backend clients
        self.colab_client: Optional[ColabClient] = None
        self.local_models: Dict[str, Any] = {}
        
        # Backend status tracking
        self.backend_status: Dict[str, Dict[BackendType, BackendStatus]] = {
            "vision": {
                BackendType.COLAB: BackendStatus(BackendType.COLAB, False),
                BackendType.LOCAL: BackendStatus(BackendType.LOCAL, True)  # Assume local is available
            },
            "nutrition": {
                BackendType.COLAB: BackendStatus(BackendType.COLAB, False),
                BackendType.LOCAL: BackendStatus(BackendType.LOCAL, True)
            },
            "cuisine": {
                BackendType.COLAB: BackendStatus(BackendType.COLAB, False),
                BackendType.LOCAL: BackendStatus(BackendType.LOCAL, True)
            }
        }
        
        # Session tracking
        self.colab_session_start: Optional[float] = None
        self.last_health_check: float = 0.0
        
        # Statistics
        self.switching_stats = {
            "total_requests": 0,
            "colab_requests": 0,
            "local_requests": 0,
            "fallback_switches": 0,
            "session_expirations": 0
        }
        
        logger.info(f"Model Backend Switcher initialized with strategy: {switching_strategy.value}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize backend clients and models."""
        try:
            # Initialize Colab client if needed
            if self.switching_strategy in [SwitchingStrategy.COLAB_FIRST, SwitchingStrategy.COLAB_ONLY]:
                self.colab_client = ColabClient()
                await self.colab_client._ensure_session()
                logger.info("Colab client initialized")
            
            # Initialize local models if needed
            if self.switching_strategy in [SwitchingStrategy.LOCAL_FIRST, SwitchingStrategy.LOCAL_ONLY, SwitchingStrategy.COLAB_FIRST]:
                await self._initialize_local_models()
                logger.info("Local models initialized")
            
            # Perform initial health check
            await self._perform_health_check()
            
        except Exception as e:
            logger.error(f"Failed to initialize backend switcher: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.colab_client:
            await self.colab_client.close()
            logger.debug("Colab client closed")
    
    async def _initialize_local_models(self):
        """Initialize local model instances."""
        try:
            # Initialize vision model
            if "vision" not in self.local_models:
                self.local_models["vision"] = VisionModel(device=self.device)
                logger.debug("Local vision model initialized")
            
            # Initialize nutrition model
            if "nutrition" not in self.local_models:
                self.local_models["nutrition"] = NutritionLLM(device=self.device)
                logger.debug("Local nutrition model initialized")
            
            # Initialize cuisine classifier
            if "cuisine" not in self.local_models:
                self.local_models["cuisine"] = CuisineClassifier(device=self.device)
                logger.debug("Local cuisine classifier initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize local models: {e}")
            # Mark local models as unavailable
            for model_type in ["vision", "nutrition", "cuisine"]:
                self.backend_status[model_type][BackendType.LOCAL].is_available = False
            raise
    
    def register_colab_endpoint(self, model_type: str, ngrok_url: str, 
                               endpoint_path: str = "/predict") -> bool:
        """
        Register a Colab endpoint for a specific model type.
        
        Args:
            model_type: Type of model ("vision", "nutrition", "cuisine")
            ngrok_url: Base ngrok URL
            endpoint_path: API endpoint path
            
        Returns:
            True if registration successful, False otherwise
        """
        if not self.colab_client:
            logger.error("Colab client not initialized")
            return False
        
        try:
            # Map string to enum
            colab_model_type = ColabModelType(model_type)
            
            success = self.colab_client.register_endpoint(
                colab_model_type, ngrok_url, endpoint_path
            )
            
            if success:
                # Mark Colab backend as potentially available
                self.backend_status[model_type][BackendType.COLAB].is_available = True
                self.colab_session_start = time.time()
                logger.info(f"Registered Colab endpoint for {model_type}: {ngrok_url}")
            
            return success
            
        except ValueError:
            logger.error(f"Invalid model type: {model_type}")
            return False
        except Exception as e:
            logger.error(f"Failed to register Colab endpoint: {e}")
            return False
    
    async def _perform_health_check(self):
        """Perform health check on all backends."""
        current_time = time.time()
        
        # Skip if too soon since last check
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        logger.debug("Performing backend health check")
        
        # Check Colab backends
        if self.colab_client:
            try:
                health_results = await self.colab_client.health_check()
                
                for colab_model_type, status in health_results.items():
                    model_type = colab_model_type.value
                    is_healthy = status == ColabConnectionStatus.CONNECTED
                    
                    self.backend_status[model_type][BackendType.COLAB].is_available = is_healthy
                    
                    if is_healthy:
                        self.backend_status[model_type][BackendType.COLAB].last_success = current_time
                        self.backend_status[model_type][BackendType.COLAB].consecutive_failures = 0
                    else:
                        self.backend_status[model_type][BackendType.COLAB].last_failure = current_time
                        self.backend_status[model_type][BackendType.COLAB].consecutive_failures += 1
                
            except Exception as e:
                logger.warning(f"Colab health check failed: {e}")
                # Mark all Colab backends as unavailable
                for model_type in ["vision", "nutrition", "cuisine"]:
                    self.backend_status[model_type][BackendType.COLAB].is_available = False
        
        # Check for session expiration
        if self.colab_session_start and (current_time - self.colab_session_start) > self.session_timeout:
            logger.warning("Colab session may have expired")
            self.switching_stats["session_expirations"] += 1
            # Mark Colab backends as potentially unavailable
            for model_type in ["vision", "nutrition", "cuisine"]:
                self.backend_status[model_type][BackendType.COLAB].is_available = False
        
        self.last_health_check = current_time
    
    def _choose_backend(self, model_type: str) -> Optional[BackendType]:
        """
        Choose the best available backend for a model type.
        
        Args:
            model_type: Type of model ("vision", "nutrition", "cuisine")
            
        Returns:
            BackendType to use, or None if none available
        """
        colab_status = self.backend_status[model_type][BackendType.COLAB]
        local_status = self.backend_status[model_type][BackendType.LOCAL]
        
        # Apply switching strategy
        if self.switching_strategy == SwitchingStrategy.COLAB_ONLY:
            return BackendType.COLAB if colab_status.is_available else None
        
        elif self.switching_strategy == SwitchingStrategy.LOCAL_ONLY:
            return BackendType.LOCAL if local_status.is_available else None
        
        elif self.switching_strategy == SwitchingStrategy.COLAB_FIRST:
            # Check if Colab has too many consecutive failures
            if (colab_status.is_available and 
                colab_status.consecutive_failures < self.failure_threshold):
                return BackendType.COLAB
            elif local_status.is_available:
                return BackendType.LOCAL
            else:
                return None
        
        elif self.switching_strategy == SwitchingStrategy.LOCAL_FIRST:
            if local_status.is_available:
                return BackendType.LOCAL
            elif (colab_status.is_available and 
                  colab_status.consecutive_failures < self.failure_threshold):
                return BackendType.COLAB
            else:
                return None
        
        return None
    
    async def _execute_colab_request(self, model_type: str, input_data: Dict[str, Any]) -> ModelResponse:
        """
        Execute request using Colab backend.
        
        Args:
            model_type: Type of model
            input_data: Input data for the model
            
        Returns:
            ModelResponse with results
        """
        if not self.colab_client:
            return ModelResponse(
                success=False,
                error_message="Colab client not initialized",
                backend_used=BackendType.COLAB
            )
        
        try:
            start_time = time.time()
            
            # Make request based on model type
            if model_type == "vision":
                image_url = input_data.get("image_url")
                if not image_url:
                    raise ValueError("image_url required for vision model")
                response = await self.colab_client.predict_vision(image_url)
                
            elif model_type == "nutrition":
                ingredients = input_data.get("ingredients", [])
                description = input_data.get("description", "")
                response = await self.colab_client.predict_nutrition(ingredients, description)
                
            elif model_type == "cuisine":
                ingredients = input_data.get("ingredients", [])
                description = input_data.get("description", "")
                response = await self.colab_client.predict_cuisine(ingredients, description)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            response_time = time.time() - start_time
            
            # Update backend status
            backend_status = self.backend_status[model_type][BackendType.COLAB]
            backend_status.total_requests += 1
            
            if response.success:
                backend_status.successful_requests += 1
                backend_status.last_success = time.time()
                backend_status.consecutive_failures = 0
                
                # Update average response time
                if backend_status.successful_requests == 1:
                    backend_status.average_response_time = response_time
                else:
                    backend_status.average_response_time = (
                        (backend_status.average_response_time * (backend_status.successful_requests - 1) + response_time) /
                        backend_status.successful_requests
                    )
                
                return ModelResponse(
                    success=True,
                    data=response.data,
                    backend_used=BackendType.COLAB,
                    response_time=response_time,
                    retry_count=response.retry_count
                )
            else:
                backend_status.last_failure = time.time()
                backend_status.consecutive_failures += 1
                
                return ModelResponse(
                    success=False,
                    error_message=response.error_message,
                    backend_used=BackendType.COLAB,
                    response_time=response_time,
                    retry_count=response.retry_count
                )
                
        except Exception as e:
            # Update failure status
            backend_status = self.backend_status[model_type][BackendType.COLAB]
            backend_status.last_failure = time.time()
            backend_status.consecutive_failures += 1
            
            error_msg = f"Colab request failed: {str(e)}"
            logger.error(error_msg)
            
            return ModelResponse(
                success=False,
                error_message=error_msg,
                backend_used=BackendType.COLAB
            )
    
    async def _execute_local_request(self, model_type: str, input_data: Dict[str, Any]) -> ModelResponse:
        """
        Execute request using local backend.
        
        Args:
            model_type: Type of model
            input_data: Input data for the model
            
        Returns:
            ModelResponse with results
        """
        if model_type not in self.local_models:
            return ModelResponse(
                success=False,
                error_message=f"Local {model_type} model not initialized",
                backend_used=BackendType.LOCAL
            )
        
        try:
            start_time = time.time()
            
            # Execute based on model type
            if model_type == "vision":
                image_url = input_data.get("image_url")
                if not image_url:
                    raise ValueError("image_url required for vision model")
                
                model = self.local_models["vision"]
                result_data = model.analyze_image(image_url)
                
            elif model_type == "nutrition":
                ingredients = input_data.get("ingredients", [])
                description = input_data.get("description", "")
                
                # Create vision_results format for nutrition model
                vision_results = {
                    "ingredients": ingredients,
                    "description": description
                }
                
                model = self.local_models["nutrition"]
                result_data = model.analyze_nutrition(vision_results)
                
            elif model_type == "cuisine":
                ingredients = input_data.get("ingredients", [])
                description = input_data.get("description", "")
                
                # Create vision_results format for cuisine model
                vision_results = {
                    "ingredients": ingredients,
                    "description": description
                }
                
                model = self.local_models["cuisine"]
                result_data = model.analyze_cuisine(vision_results)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            response_time = time.time() - start_time
            
            # Update backend status
            backend_status = self.backend_status[model_type][BackendType.LOCAL]
            backend_status.total_requests += 1
            
            # Check if local execution was successful
            success = result_data.get("analysis_status") == "success" if "analysis_status" in result_data else True
            
            if success:
                backend_status.successful_requests += 1
                backend_status.last_success = time.time()
                backend_status.consecutive_failures = 0
                
                # Update average response time
                if backend_status.successful_requests == 1:
                    backend_status.average_response_time = response_time
                else:
                    backend_status.average_response_time = (
                        (backend_status.average_response_time * (backend_status.successful_requests - 1) + response_time) /
                        backend_status.successful_requests
                    )
                
                return ModelResponse(
                    success=True,
                    data=result_data,
                    backend_used=BackendType.LOCAL,
                    response_time=response_time
                )
            else:
                backend_status.last_failure = time.time()
                backend_status.consecutive_failures += 1
                
                error_msg = result_data.get("error_message", "Local model execution failed")
                return ModelResponse(
                    success=False,
                    error_message=error_msg,
                    backend_used=BackendType.LOCAL,
                    response_time=response_time
                )
                
        except Exception as e:
            # Update failure status
            backend_status = self.backend_status[model_type][BackendType.LOCAL]
            backend_status.last_failure = time.time()
            backend_status.consecutive_failures += 1
            
            error_msg = f"Local model execution failed: {str(e)}"
            logger.error(error_msg)
            
            return ModelResponse(
                success=False,
                error_message=error_msg,
                backend_used=BackendType.LOCAL
            )
    
    async def predict(self, model_type: str, input_data: Dict[str, Any]) -> ModelResponse:
        """
        Make prediction using the best available backend with automatic fallback.
        
        Args:
            model_type: Type of model ("vision", "nutrition", "cuisine")
            input_data: Input data for the model
            
        Returns:
            ModelResponse with prediction results
        """
        # Update statistics
        self.switching_stats["total_requests"] += 1
        
        # Perform health check if needed
        await self._perform_health_check()
        
        # Choose primary backend
        primary_backend = self._choose_backend(model_type)
        
        if not primary_backend:
            return ModelResponse(
                success=False,
                error_message=f"No available backend for {model_type} model"
            )
        
        # Try primary backend
        logger.debug(f"Using {primary_backend.value} backend for {model_type}")
        
        if primary_backend == BackendType.COLAB:
            self.switching_stats["colab_requests"] += 1
            response = await self._execute_colab_request(model_type, input_data)
        else:
            self.switching_stats["local_requests"] += 1
            response = await self._execute_local_request(model_type, input_data)
        
        # If primary backend failed, try fallback
        if not response.success and self.switching_strategy in [SwitchingStrategy.COLAB_FIRST, SwitchingStrategy.LOCAL_FIRST]:
            logger.warning(f"Primary backend ({primary_backend.value}) failed for {model_type}, trying fallback")
            
            # Determine fallback backend
            if primary_backend == BackendType.COLAB:
                fallback_backend = BackendType.LOCAL
                if self.backend_status[model_type][BackendType.LOCAL].is_available:
                    logger.info(f"Falling back to local backend for {model_type}")
                    self.switching_stats["local_requests"] += 1
                    self.switching_stats["fallback_switches"] += 1
                    fallback_response = await self._execute_local_request(model_type, input_data)
                    if fallback_response.success:
                        fallback_response.fallback_used = True
                        return fallback_response
            else:
                fallback_backend = BackendType.COLAB
                if self.backend_status[model_type][BackendType.COLAB].is_available:
                    logger.info(f"Falling back to Colab backend for {model_type}")
                    self.switching_stats["colab_requests"] += 1
                    self.switching_stats["fallback_switches"] += 1
                    fallback_response = await self._execute_colab_request(model_type, input_data)
                    if fallback_response.success:
                        fallback_response.fallback_used = True
                        return fallback_response
        
        return response
    
    async def predict_vision(self, image_url: str) -> ModelResponse:
        """
        Predict using vision model with automatic backend selection.
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            ModelResponse with vision analysis results
        """
        input_data = {"image_url": image_url}
        return await self.predict("vision", input_data)
    
    async def predict_nutrition(self, ingredients: List[str], description: str) -> ModelResponse:
        """
        Predict using nutrition model with automatic backend selection.
        
        Args:
            ingredients: List of food ingredients
            description: Food description
            
        Returns:
            ModelResponse with nutrition analysis results
        """
        input_data = {
            "ingredients": ingredients,
            "description": description
        }
        return await self.predict("nutrition", input_data)
    
    async def predict_cuisine(self, ingredients: List[str], description: str) -> ModelResponse:
        """
        Predict using cuisine model with automatic backend selection.
        
        Args:
            ingredients: List of food ingredients
            description: Food description
            
        Returns:
            ModelResponse with cuisine classification results
        """
        input_data = {
            "ingredients": ingredients,
            "description": description
        }
        return await self.predict("cuisine", input_data)
    
    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get comprehensive backend status information.
        
        Returns:
            Dictionary with backend status and statistics
        """
        status = {
            "switching_strategy": self.switching_strategy.value,
            "session_timeout": self.session_timeout,
            "failure_threshold": self.failure_threshold,
            "colab_session_age": None,
            "backends": {},
            "statistics": self.switching_stats.copy()
        }
        
        # Add session age if Colab is active
        if self.colab_session_start:
            status["colab_session_age"] = time.time() - self.colab_session_start
        
        # Add backend status for each model type
        for model_type in ["vision", "nutrition", "cuisine"]:
            status["backends"][model_type] = {}
            
            for backend_type in [BackendType.COLAB, BackendType.LOCAL]:
                backend_status = self.backend_status[model_type][backend_type]
                
                status["backends"][model_type][backend_type.value] = {
                    "is_available": backend_status.is_available,
                    "last_success": backend_status.last_success,
                    "last_failure": backend_status.last_failure,
                    "consecutive_failures": backend_status.consecutive_failures,
                    "average_response_time": backend_status.average_response_time,
                    "total_requests": backend_status.total_requests,
                    "successful_requests": backend_status.successful_requests,
                    "success_rate": (
                        backend_status.successful_requests / backend_status.total_requests
                        if backend_status.total_requests > 0 else 0.0
                    )
                }
        
        # Calculate overall statistics
        total_requests = status["statistics"]["total_requests"]
        if total_requests > 0:
            status["statistics"]["colab_usage_rate"] = status["statistics"]["colab_requests"] / total_requests
            status["statistics"]["local_usage_rate"] = status["statistics"]["local_requests"] / total_requests
            status["statistics"]["fallback_rate"] = status["statistics"]["fallback_switches"] / total_requests
        else:
            status["statistics"]["colab_usage_rate"] = 0.0
            status["statistics"]["local_usage_rate"] = 0.0
            status["statistics"]["fallback_rate"] = 0.0
        
        return status
    
    def set_switching_strategy(self, strategy: SwitchingStrategy):
        """
        Update the switching strategy.
        
        Args:
            strategy: New switching strategy
        """
        old_strategy = self.switching_strategy
        self.switching_strategy = strategy
        logger.info(f"Switching strategy changed from {old_strategy.value} to {strategy.value}")
    
    async def force_backend_refresh(self):
        """Force refresh of all backend statuses."""
        logger.info("Forcing backend refresh")
        self.last_health_check = 0.0  # Force health check
        await self._perform_health_check()


# Convenience functions
async def create_backend_switcher(
    switching_strategy: SwitchingStrategy = SwitchingStrategy.COLAB_FIRST,
    device: Optional[str] = None
) -> ModelBackendSwitcher:
    """
    Create and initialize a model backend switcher.
    
    Args:
        switching_strategy: Strategy for backend selection
        device: Device for local models
        
    Returns:
        Initialized ModelBackendSwitcher
    """
    switcher = ModelBackendSwitcher(switching_strategy=switching_strategy, device=device)
    await switcher.initialize()
    return switcher


async def test_backend_switching(ngrok_urls: Dict[str, str]) -> Dict[str, Any]:
    """
    Test backend switching functionality with provided ngrok URLs.
    
    Args:
        ngrok_urls: Dictionary mapping model types to ngrok URLs
        
    Returns:
        Test results dictionary
    """
    async with ModelBackendSwitcher(SwitchingStrategy.COLAB_FIRST) as switcher:
        # Register Colab endpoints
        for model_type, url in ngrok_urls.items():
            switcher.register_colab_endpoint(model_type, url)
        
        # Test each model type
        results = {}
        
        # Test vision
        if "vision" in ngrok_urls:
            response = await switcher.predict_vision("https://example.com/test_image.jpg")
            results["vision"] = {
                "success": response.success,
                "backend_used": response.backend_used.value if response.backend_used else None,
                "fallback_used": response.fallback_used
            }
        
        # Test nutrition
        if "nutrition" in ngrok_urls:
            response = await switcher.predict_nutrition(["chicken", "rice"], "Chicken and rice dish")
            results["nutrition"] = {
                "success": response.success,
                "backend_used": response.backend_used.value if response.backend_used else None,
                "fallback_used": response.fallback_used
            }
        
        # Test cuisine
        if "cuisine" in ngrok_urls:
            response = await switcher.predict_cuisine(["pasta", "tomato"], "Italian pasta dish")
            results["cuisine"] = {
                "success": response.success,
                "backend_used": response.backend_used.value if response.backend_used else None,
                "fallback_used": response.fallback_used
            }
        
        # Add backend status
        results["backend_status"] = switcher.get_backend_status()
        
        return results