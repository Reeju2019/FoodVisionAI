"""
Google Colab Client for FoodVisionAI

Implements remote model inference through Google Colab endpoints via ngrok tunnels.
Provides API calls to Colab-hosted models with timeout and retry logic for network issues.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import backoff
from urllib.parse import urljoin, urlparse


class ColabModelType(Enum):
    """Types of models available on Colab."""
    VISION = "vision"
    NUTRITION = "nutrition"
    CUISINE = "cuisine"


class ColabConnectionStatus(Enum):
    """Connection status for Colab endpoints."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ColabEndpoint:
    """Configuration for a Colab model endpoint."""
    model_type: ColabModelType
    ngrok_url: str
    endpoint_path: str
    timeout_seconds: int = 30
    max_retries: int = 3
    is_active: bool = True
    last_health_check: Optional[float] = None
    health_status: ColabConnectionStatus = ColabConnectionStatus.UNKNOWN


@dataclass
class ColabResponse:
    """Response from Colab model inference."""
    success: bool
    data: Optional[Dict] = None
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    endpoint_url: Optional[str] = None
    retry_count: int = 0


class ColabClient:
    """
    Client for communicating with Google Colab model endpoints.
    
    Handles ngrok tunnel communication, API calls with timeout and retry logic,
    and automatic fallback detection for network issues.
    """
    
    def __init__(self, session_timeout: int = 60, health_check_interval: int = 300):
        """
        Initialize the Colab Client.
        
        Args:
            session_timeout: Default timeout for HTTP sessions in seconds
            health_check_interval: Interval between health checks in seconds
        """
        self.session_timeout = session_timeout
        self.health_check_interval = health_check_interval
        
        # Endpoint configurations
        self.endpoints: Dict[ColabModelType, ColabEndpoint] = {}
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Connection tracking
        self.last_global_health_check = 0.0
        self.connection_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "timeout_requests": 0,
            "retry_requests": 0
        }
        
        logger.info("Colab Client initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.session_timeout)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": "FoodVisionAI-ColabClient/1.0",
                    "Content-Type": "application/json"
                }
            )
            logger.debug("HTTP session created")
    
    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("HTTP session closed")
    
    def register_endpoint(self, model_type: ColabModelType, ngrok_url: str, 
                         endpoint_path: str = "/predict", timeout_seconds: int = 30,
                         max_retries: int = 3) -> bool:
        """
        Register a Colab model endpoint.
        
        Args:
            model_type: Type of model (vision, nutrition, cuisine)
            ngrok_url: Base ngrok URL (e.g., "https://abc123.ngrok.io")
            endpoint_path: API endpoint path (e.g., "/predict")
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retries
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate ngrok URL
            parsed_url = urlparse(ngrok_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.error(f"Invalid ngrok URL: {ngrok_url}")
                return False
            
            # Create endpoint configuration
            endpoint = ColabEndpoint(
                model_type=model_type,
                ngrok_url=ngrok_url.rstrip('/'),
                endpoint_path=endpoint_path,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                is_active=True
            )
            
            self.endpoints[model_type] = endpoint
            
            logger.info(f"Registered {model_type.value} endpoint: {ngrok_url}{endpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register endpoint for {model_type.value}: {e}")
            return False
    
    def unregister_endpoint(self, model_type: ColabModelType) -> bool:
        """
        Unregister a Colab model endpoint.
        
        Args:
            model_type: Type of model to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if model_type in self.endpoints:
            del self.endpoints[model_type]
            logger.info(f"Unregistered {model_type.value} endpoint")
            return True
        else:
            logger.warning(f"No endpoint registered for {model_type.value}")
            return False
    
    def get_endpoint_url(self, model_type: ColabModelType) -> Optional[str]:
        """
        Get full endpoint URL for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Full endpoint URL or None if not registered
        """
        if model_type not in self.endpoints:
            return None
        
        endpoint = self.endpoints[model_type]
        return urljoin(endpoint.ngrok_url + '/', endpoint.endpoint_path.lstrip('/'))
    
    async def health_check(self, model_type: Optional[ColabModelType] = None) -> Dict[ColabModelType, ColabConnectionStatus]:
        """
        Perform health check on Colab endpoints.
        
        Args:
            model_type: Specific model type to check, or None for all
            
        Returns:
            Dictionary mapping model types to their connection status
        """
        await self._ensure_session()
        
        # Determine which endpoints to check
        if model_type is not None:
            endpoints_to_check = {model_type: self.endpoints[model_type]} if model_type in self.endpoints else {}
        else:
            endpoints_to_check = self.endpoints.copy()
        
        results = {}
        
        for mt, endpoint in endpoints_to_check.items():
            try:
                # Create health check URL (typically /health or /status)
                health_url = urljoin(endpoint.ngrok_url + '/', 'health')
                
                start_time = time.time()
                
                async with self.session.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=10)  # Short timeout for health checks
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        status = ColabConnectionStatus.CONNECTED
                        logger.debug(f"{mt.value} endpoint healthy ({response_time:.2f}s)")
                    else:
                        status = ColabConnectionStatus.ERROR
                        logger.warning(f"{mt.value} endpoint returned status {response.status}")
                
                # Update endpoint status
                endpoint.health_status = status
                endpoint.last_health_check = time.time()
                
                results[mt] = status
                
            except asyncio.TimeoutError:
                status = ColabConnectionStatus.TIMEOUT
                endpoint.health_status = status
                endpoint.last_health_check = time.time()
                results[mt] = status
                logger.warning(f"{mt.value} endpoint health check timed out")
                
            except Exception as e:
                status = ColabConnectionStatus.ERROR
                endpoint.health_status = status
                endpoint.last_health_check = time.time()
                results[mt] = status
                logger.error(f"{mt.value} endpoint health check failed: {e}")
        
        # Update global health check time
        self.last_global_health_check = time.time()
        
        return results
    
    async def _should_perform_health_check(self, model_type: ColabModelType) -> bool:
        """
        Determine if a health check should be performed before making a request.
        
        Args:
            model_type: Model type to check
            
        Returns:
            True if health check should be performed
        """
        if model_type not in self.endpoints:
            return False
        
        endpoint = self.endpoints[model_type]
        current_time = time.time()
        
        # Perform health check if:
        # 1. Never checked before
        # 2. Last check was too long ago
        # 3. Last check indicated problems
        
        if endpoint.last_health_check is None:
            return True
        
        time_since_check = current_time - endpoint.last_health_check
        if time_since_check > self.health_check_interval:
            return True
        
        if endpoint.health_status in [ColabConnectionStatus.ERROR, ColabConnectionStatus.TIMEOUT]:
            # More frequent checks for problematic endpoints
            if time_since_check > 60:  # Check every minute for failed endpoints
                return True
        
        return False
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60,
        jitter=backoff.random_jitter
    )
    async def _make_request(self, url: str, payload: Dict, timeout: int) -> Tuple[bool, Dict, float]:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: Request URL
            payload: JSON payload
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (success, response_data, response_time)
        """
        await self._ensure_session()
        
        start_time = time.time()
        
        async with self.session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            response_time = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                return True, data, response_time
            else:
                error_text = await response.text()
                logger.error(f"HTTP {response.status}: {error_text}")
                return False, {"error": f"HTTP {response.status}: {error_text}"}, response_time
    
    async def predict(self, model_type: ColabModelType, input_data: Dict) -> ColabResponse:
        """
        Make prediction request to Colab model endpoint.
        
        Args:
            model_type: Type of model to call
            input_data: Input data for the model
            
        Returns:
            ColabResponse with prediction results
        """
        # Update stats
        self.connection_stats["total_requests"] += 1
        
        # Check if endpoint is registered
        if model_type not in self.endpoints:
            error_msg = f"No endpoint registered for {model_type.value}"
            logger.error(error_msg)
            self.connection_stats["failed_requests"] += 1
            return ColabResponse(
                success=False,
                error_message=error_msg
            )
        
        endpoint = self.endpoints[model_type]
        
        # Check if endpoint is active
        if not endpoint.is_active:
            error_msg = f"Endpoint for {model_type.value} is marked as inactive"
            logger.warning(error_msg)
            self.connection_stats["failed_requests"] += 1
            return ColabResponse(
                success=False,
                error_message=error_msg
            )
        
        # Perform health check if needed
        if await self._should_perform_health_check(model_type):
            logger.debug(f"Performing health check for {model_type.value}")
            health_results = await self.health_check(model_type)
            
            if health_results.get(model_type) != ColabConnectionStatus.CONNECTED:
                error_msg = f"Health check failed for {model_type.value} endpoint"
                logger.warning(error_msg)
                self.connection_stats["failed_requests"] += 1
                return ColabResponse(
                    success=False,
                    error_message=error_msg
                )
        
        # Get endpoint URL
        url = self.get_endpoint_url(model_type)
        if not url:
            error_msg = f"Could not construct URL for {model_type.value}"
            logger.error(error_msg)
            self.connection_stats["failed_requests"] += 1
            return ColabResponse(
                success=False,
                error_message=error_msg
            )
        
        # Prepare payload
        payload = {
            "model_type": model_type.value,
            "input_data": input_data,
            "timestamp": time.time()
        }
        
        retry_count = 0
        last_error = None
        
        # Retry loop
        for attempt in range(endpoint.max_retries + 1):
            try:
                logger.debug(f"Making request to {model_type.value} endpoint (attempt {attempt + 1})")
                
                success, response_data, response_time = await self._make_request(
                    url, payload, endpoint.timeout_seconds
                )
                
                if success:
                    self.connection_stats["successful_requests"] += 1
                    if retry_count > 0:
                        self.connection_stats["retry_requests"] += 1
                    
                    logger.info(f"{model_type.value} prediction successful ({response_time:.2f}s)")
                    
                    return ColabResponse(
                        success=True,
                        data=response_data,
                        response_time=response_time,
                        endpoint_url=url,
                        retry_count=retry_count
                    )
                else:
                    last_error = response_data.get("error", "Unknown error")
                    logger.warning(f"{model_type.value} request failed: {last_error}")
                
            except asyncio.TimeoutError:
                last_error = f"Request timed out after {endpoint.timeout_seconds}s"
                logger.warning(f"{model_type.value} request timed out (attempt {attempt + 1})")
                self.connection_stats["timeout_requests"] += 1
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"{model_type.value} request failed: {e}")
            
            # Increment retry count for next attempt
            if attempt < endpoint.max_retries:
                retry_count += 1
                # Exponential backoff delay
                delay = min(2 ** attempt, 10)  # Cap at 10 seconds
                logger.debug(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        # All retries exhausted
        self.connection_stats["failed_requests"] += 1
        error_msg = f"All {endpoint.max_retries + 1} attempts failed. Last error: {last_error}"
        
        logger.error(f"{model_type.value} prediction failed: {error_msg}")
        
        return ColabResponse(
            success=False,
            error_message=error_msg,
            endpoint_url=url,
            retry_count=retry_count
        )
    
    async def predict_vision(self, image_url: str) -> ColabResponse:
        """
        Make vision model prediction request.
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            ColabResponse with vision analysis results
        """
        input_data = {
            "image_url": image_url,
            "task": "food_analysis"
        }
        
        return await self.predict(ColabModelType.VISION, input_data)
    
    async def predict_nutrition(self, ingredients: List[str], description: str) -> ColabResponse:
        """
        Make nutrition model prediction request.
        
        Args:
            ingredients: List of food ingredients
            description: Food description
            
        Returns:
            ColabResponse with nutrition analysis results
        """
        input_data = {
            "ingredients": ingredients,
            "description": description,
            "task": "nutrition_analysis"
        }
        
        return await self.predict(ColabModelType.NUTRITION, input_data)
    
    async def predict_cuisine(self, ingredients: List[str], description: str) -> ColabResponse:
        """
        Make cuisine classification prediction request.
        
        Args:
            ingredients: List of food ingredients
            description: Food description
            
        Returns:
            ColabResponse with cuisine classification results
        """
        input_data = {
            "ingredients": ingredients,
            "description": description,
            "task": "cuisine_classification"
        }
        
        return await self.predict(ColabModelType.CUISINE, input_data)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics and endpoint status.
        
        Returns:
            Dictionary with connection statistics
        """
        stats = self.connection_stats.copy()
        
        # Calculate success rate
        total = stats["total_requests"]
        if total > 0:
            stats["success_rate"] = stats["successful_requests"] / total
            stats["failure_rate"] = stats["failed_requests"] / total
            stats["timeout_rate"] = stats["timeout_requests"] / total
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
            stats["timeout_rate"] = 0.0
        
        # Add endpoint information
        stats["endpoints"] = {}
        for model_type, endpoint in self.endpoints.items():
            stats["endpoints"][model_type.value] = {
                "url": self.get_endpoint_url(model_type),
                "is_active": endpoint.is_active,
                "health_status": endpoint.health_status.value,
                "last_health_check": endpoint.last_health_check,
                "timeout_seconds": endpoint.timeout_seconds,
                "max_retries": endpoint.max_retries
            }
        
        return stats
    
    def set_endpoint_active(self, model_type: ColabModelType, active: bool) -> bool:
        """
        Set endpoint active/inactive status.
        
        Args:
            model_type: Model type
            active: Whether endpoint should be active
            
        Returns:
            True if status was updated, False if endpoint not found
        """
        if model_type in self.endpoints:
            self.endpoints[model_type].is_active = active
            logger.info(f"Set {model_type.value} endpoint {'active' if active else 'inactive'}")
            return True
        else:
            logger.warning(f"No endpoint found for {model_type.value}")
            return False
    
    async def test_all_endpoints(self) -> Dict[ColabModelType, bool]:
        """
        Test all registered endpoints with sample data.
        
        Returns:
            Dictionary mapping model types to test success status
        """
        results = {}
        
        # Test vision endpoint
        if ColabModelType.VISION in self.endpoints:
            try:
                response = await self.predict_vision("https://example.com/test_image.jpg")
                results[ColabModelType.VISION] = response.success
            except Exception as e:
                logger.error(f"Vision endpoint test failed: {e}")
                results[ColabModelType.VISION] = False
        
        # Test nutrition endpoint
        if ColabModelType.NUTRITION in self.endpoints:
            try:
                response = await self.predict_nutrition(["chicken", "rice"], "Chicken and rice dish")
                results[ColabModelType.NUTRITION] = response.success
            except Exception as e:
                logger.error(f"Nutrition endpoint test failed: {e}")
                results[ColabModelType.NUTRITION] = False
        
        # Test cuisine endpoint
        if ColabModelType.CUISINE in self.endpoints:
            try:
                response = await self.predict_cuisine(["pasta", "tomato"], "Italian pasta dish")
                results[ColabModelType.CUISINE] = response.success
            except Exception as e:
                logger.error(f"Cuisine endpoint test failed: {e}")
                results[ColabModelType.CUISINE] = False
        
        return results


# Convenience functions for quick usage
async def create_colab_client() -> ColabClient:
    """Create and return a new Colab client."""
    client = ColabClient()
    await client._ensure_session()
    return client


async def test_colab_connection(ngrok_url: str, model_type: ColabModelType) -> bool:
    """
    Quick test of Colab connection.
    
    Args:
        ngrok_url: Base ngrok URL
        model_type: Type of model to test
        
    Returns:
        True if connection successful, False otherwise
    """
    async with ColabClient() as client:
        client.register_endpoint(model_type, ngrok_url)
        health_results = await client.health_check(model_type)
        return health_results.get(model_type) == ColabConnectionStatus.CONNECTED