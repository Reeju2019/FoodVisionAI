"""
Custom middleware for FastAPI application.
"""
import time
import logging
import traceback
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("api.requests")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response information."""
        start_time = time.time()
        
        # Log request
        self.logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        self.logger.info(
            f"Response: {response.status_code} "
            f"processed in {process_time:.3f}s"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("api.errors")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle exceptions and return appropriate error responses."""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions (they're handled by FastAPI)
            raise
            
        except Exception as e:
            # Log the full exception
            self.logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "path": str(request.url.path),
                    "method": request.method
                }
            )