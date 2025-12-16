"""
Health check and monitoring endpoints for FoodVisionAI.
"""
import time
import psutil
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..database import get_database
from ..config import settings

router = APIRouter()

# Track application start time
START_TIME = time.time()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    Returns 200 if the application is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "FoodVisionAI"
    }


@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> Dict[str, Any]:
    """
    Detailed health check with component status.
    Checks database connectivity and system resources.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime_seconds": int(time.time() - START_TIME),
        "components": {}
    }
    
    # Check database connectivity
    try:
        await db.command("ping")
        health_status["components"]["database"] = {
            "status": "healthy",
            "type": "mongodb",
            "message": "Connected"
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "type": "mongodb",
            "message": f"Connection failed: {str(e)}"
        }
    
    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status["components"]["system"] = {
            "status": "healthy",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024 * 1024 * 1024)
        }
        
        # Mark as unhealthy if resources are critically low
        if memory.percent > 90 or disk.percent > 90:
            health_status["status"] = "degraded"
            health_status["components"]["system"]["status"] = "degraded"
            health_status["components"]["system"]["message"] = "Low resources"
            
    except Exception as e:
        health_status["components"]["system"] = {
            "status": "unknown",
            "message": f"Could not check system resources: {str(e)}"
        }
    
    return health_status


@router.get("/health/ready")
async def readiness_check(
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> Dict[str, Any]:
    """
    Readiness check for Kubernetes/container orchestration.
    Returns 200 only if the service is ready to accept traffic.
    """
    try:
        # Check database is accessible
        await db.command("ping")
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": str(e)
        }


@router.get("/health/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for Kubernetes/container orchestration.
    Returns 200 if the application process is alive.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - START_TIME)
    }


@router.get("/metrics")
async def metrics(
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> Dict[str, Any]:
    """
    Application metrics endpoint.
    Returns various metrics for monitoring.
    """
    metrics_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - START_TIME),
        "application": {
            "name": settings.app_name,
            "version": "1.0.0",
            "debug": settings.debug
        }
    }
    
    # Database metrics
    try:
        db_stats = await db.command("dbStats")
        metrics_data["database"] = {
            "collections": db_stats.get("collections", 0),
            "objects": db_stats.get("objects", 0),
            "data_size_mb": db_stats.get("dataSize", 0) / (1024 * 1024),
            "storage_size_mb": db_stats.get("storageSize", 0) / (1024 * 1024)
        }
    except Exception as e:
        metrics_data["database"] = {
            "error": str(e)
        }
    
    # System metrics
    try:
        metrics_data["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except Exception:
        pass
    
    return metrics_data

