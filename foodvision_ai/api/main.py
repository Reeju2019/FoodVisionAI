"""
Main FastAPI application with CORS and middleware configuration.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..database import db_connection, get_database, DatabaseOperations
from ..database.models import AnalysisStatusResponse
from .endpoints import upload_router, status_router, analytics_router
from .middleware import LoggingMiddleware, ErrorHandlingMiddleware
from .health import router as health_router
from .security import RateLimitMiddleware, SecurityHeadersMiddleware, InputValidationMiddleware
from ..config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logging.info("Starting FoodVisionAI application...")
    
    # Connect to database
    connected = await db_connection.connect()
    if not connected:
        logging.warning("Failed to connect to database - running in no-database mode")
    else:
        logging.info("Database connected successfully")
    
    yield
    
    # Shutdown
    logging.info("Shutting down FoodVisionAI application...")
    await db_connection.disconnect()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="FoodVisionAI",
        description="Automated nutritional analysis application using AI pipeline",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configure CORS - use environment variable for production
    allowed_origins = settings.cors_origins.split(",") if hasattr(settings, 'cors_origins') and settings.cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Add trusted host middleware for security
    allowed_hosts = settings.allowed_hosts.split(",") if hasattr(settings, 'allowed_hosts') and settings.allowed_hosts else ["*"]
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )

    # Add security middleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(InputValidationMiddleware)

    # Add rate limiting (only in production)
    if not settings.debug:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=100, burst=20)

    # Add custom middleware
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Include health check router (no prefix)
    app.include_router(health_router, tags=["health"])

    # Include routers with both prefixed and non-prefixed routes for compatibility
    app.include_router(upload_router, prefix="/api/v1", tags=["upload"])
    app.include_router(status_router, prefix="/api/v1", tags=["status"])
    app.include_router(analytics_router, prefix="/api/v1", tags=["analytics"])

    # Also include routers without prefix for frontend compatibility
    app.include_router(upload_router, tags=["upload-compat"])
    app.include_router(status_router, tags=["status-compat"])
    app.include_router(analytics_router, tags=["analytics-compat"])
    
    # Serve static frontend files
    from pathlib import Path
    frontend_path = Path("frontend")
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory="frontend"), name="static")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            db = await get_database()
            if db is not None:
                # Simple database ping
                await db.command("ping")
                return {"status": "healthy", "database": "connected"}
            else:
                return {"status": "healthy", "database": "not_connected", "note": "Running in no-database mode"}
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return {"status": "degraded", "database": "error", "error": str(e)}
    
    # Root endpoint - redirect to upload page
    @app.get("/")
    async def root():
        """Root endpoint - redirect to upload page."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/upload", status_code=302)
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "foodvision_ai.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )