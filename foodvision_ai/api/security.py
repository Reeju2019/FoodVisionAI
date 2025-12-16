"""
Security middleware and utilities for FoodVisionAI.
"""
import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent abuse.
    
    Implements a simple token bucket algorithm per IP address.
    """
    
    def __init__(self, app, requests_per_minute: int = 60, burst: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.clients: Dict[str, Tuple[float, int]] = defaultdict(lambda: (time.time(), burst))
        self.cleanup_interval = 300  # Clean up old entries every 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Check rate limit
        current_time = time.time()
        last_request_time, tokens = self.clients[client_ip]
        
        # Refill tokens based on time passed
        time_passed = current_time - last_request_time
        tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
        tokens = min(self.burst, tokens + tokens_to_add)
        
        # Check if request is allowed
        if tokens < 1:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": int((1 - tokens) / (self.requests_per_minute / 60.0))
                }
            )
        
        # Consume a token
        tokens -= 1
        self.clients[client_ip] = (current_time, tokens)
        
        # Periodic cleanup of old entries
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(tokens))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than 10 minutes."""
        cutoff_time = current_time - 600
        self.clients = {
            ip: (last_time, tokens)
            for ip, (last_time, tokens) in self.clients.items()
            if last_time > cutoff_time
        }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate and sanitize input to prevent common attacks.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Check for suspicious patterns in URL
        suspicious_patterns = [
            "../", "..\\",  # Path traversal
            "<script", "</script>",  # XSS
            "javascript:",  # XSS
            "eval(", "exec(",  # Code injection
        ]
        
        url_path = request.url.path.lower()
        for pattern in suspicious_patterns:
            if pattern in url_path:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid request"}
                )
        
        # Check content length for upload endpoints
        if request.url.path.startswith("/api/v1/upload") or request.url.path == "/upload":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": "File too large. Maximum size is 10MB."}
                )
        
        return await call_next(request)


def validate_image_file(file_content: bytes, filename: str) -> bool:
    """
    Validate that uploaded file is actually an image.
    
    Args:
        file_content: File content bytes
        filename: Original filename
        
    Returns:
        True if valid image, False otherwise
    """
    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""
    if f".{file_ext}" not in allowed_extensions:
        return False
    
    # Check magic bytes (file signature)
    if len(file_content) < 12:
        return False
    
    # JPEG magic bytes
    if file_content[:2] == b'\xff\xd8':
        return True
    
    # PNG magic bytes
    if file_content[:8] == b'\x89PNG\r\n\x1a\n':
        return True
    
    # WebP magic bytes
    if file_content[:4] == b'RIFF' and file_content[8:12] == b'WEBP':
        return True
    
    return False

