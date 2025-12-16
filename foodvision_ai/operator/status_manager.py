"""
Processing Status Management for FoodVisionAI

Implements In_Progress flag management, Is_Error flag handling,
and status finalization logic for completed processing.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from loguru import logger

from .logging_system import ModelRemarkLogger, ComponentType, LogLevel


class ProcessingPhase(Enum):
    """Processing phases for status tracking."""
    INITIALIZATION = "initialization"
    VISION_PROCESSING = "vision_processing"
    NUTRITION_PROCESSING = "nutrition_processing"
    CUISINE_PROCESSING = "cuisine_processing"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"          # Non-critical errors that don't stop processing
    MEDIUM = "medium"    # Errors that affect one model but allow others to continue
    HIGH = "high"        # Errors that affect multiple models
    CRITICAL = "critical" # Errors that stop all processing


@dataclass
class ProcessingStatus:
    """Comprehensive processing status information."""
    image_id: str
    in_progress: bool
    is_error: bool
    current_phase: ProcessingPhase
    start_time: str
    last_update: str
    completion_time: Optional[str] = None
    progress_percentage: float = 0.0
    
    # Phase-specific status
    vision_completed: bool = False
    nutrition_completed: bool = False
    cuisine_completed: bool = False
    
    # Error tracking
    error_count: int = 0
    error_details: List[Dict] = None
    last_error: Optional[str] = None
    
    # Performance metrics
    total_execution_time: Optional[float] = None
    phase_execution_times: Dict[str, float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []
        if self.phase_execution_times is None:
            self.phase_execution_times = {}
        if self.metadata is None:
            self.metadata = {}


class StatusManager:
    """
    Manages processing status flags and provides comprehensive status tracking.
    
    Handles In_Progress and Is_Error flags, tracks processing phases,
    and provides status finalization logic.
    """
    
    def __init__(self, logger_instance: Optional[ModelRemarkLogger] = None):
        """
        Initialize the Status Manager.
        
        Args:
            logger_instance: Optional ModelRemarkLogger instance
        """
        self.logger = logger_instance
        
        # Status storage
        self.status_store: Dict[str, ProcessingStatus] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Status change callbacks
        self.status_callbacks: List[Callable[[str, ProcessingStatus], None]] = []
        
        # Phase timing
        self.phase_start_times: Dict[str, Dict[str, float]] = {}
        
        logger.info("Status Manager initialized")
    
    def register_status_callback(self, callback: Callable[[str, ProcessingStatus], None]):
        """
        Register a callback to be called when status changes.
        
        Args:
            callback: Function to call with (image_id, status) when status changes
        """
        self.status_callbacks.append(callback)
    
    def _notify_status_change(self, image_id: str, status: ProcessingStatus):
        """
        Notify all registered callbacks of status change.
        
        Args:
            image_id: Image identifier
            status: Updated status
        """
        for callback in self.status_callbacks:
            try:
                callback(image_id, status)
            except Exception as e:
                logger.error(f"Status callback failed: {e}")
    
    def _log_status_change(self, image_id: str, message: str, 
                          metadata: Optional[Dict] = None):
        """
        Log status change if logger is available.
        
        Args:
            image_id: Image identifier
            message: Log message
            metadata: Optional metadata
        """
        if self.logger:
            self.logger.log_info(
                session_id=image_id,
                component=ComponentType.OPERATOR,
                message=message,
                metadata=metadata
            )
    
    def initialize_processing(self, image_id: str, metadata: Optional[Dict] = None) -> ProcessingStatus:
        """
        Initialize processing status for a new image.
        
        Args:
            image_id: Unique image identifier
            metadata: Optional metadata for the processing session
            
        Returns:
            Initial ProcessingStatus
        """
        with self._lock:
            current_time = datetime.now(timezone.utc).isoformat()
            
            status = ProcessingStatus(
                image_id=image_id,
                in_progress=True,
                is_error=False,
                current_phase=ProcessingPhase.INITIALIZATION,
                start_time=current_time,
                last_update=current_time,
                metadata=metadata or {}
            )
            
            self.status_store[image_id] = status
            self.phase_start_times[image_id] = {
                ProcessingPhase.INITIALIZATION.value: asyncio.get_event_loop().time()
            }
            
            self._log_status_change(
                image_id, 
                f"Processing initialized for image {image_id}",
                {"phase": ProcessingPhase.INITIALIZATION.value}
            )
            
            self._notify_status_change(image_id, status)
            
            logger.info(f"Processing initialized for image {image_id}")
            return status
    
    def update_phase(self, image_id: str, new_phase: ProcessingPhase, 
                    progress_percentage: Optional[float] = None) -> Optional[ProcessingStatus]:
        """
        Update the current processing phase.
        
        Args:
            image_id: Image identifier
            new_phase: New processing phase
            progress_percentage: Optional progress percentage (0-100)
            
        Returns:
            Updated ProcessingStatus or None if image_id not found
        """
        with self._lock:
            if image_id not in self.status_store:
                logger.warning(f"Attempted to update phase for unknown image {image_id}")
                return None
            
            status = self.status_store[image_id]
            old_phase = status.current_phase
            current_time = datetime.now(timezone.utc).isoformat()
            loop_time = asyncio.get_event_loop().time()
            
            # Record phase execution time
            if image_id in self.phase_start_times:
                if old_phase.value in self.phase_start_times[image_id]:
                    phase_duration = loop_time - self.phase_start_times[image_id][old_phase.value]
                    status.phase_execution_times[old_phase.value] = phase_duration
                
                # Start timing new phase
                self.phase_start_times[image_id][new_phase.value] = loop_time
            
            # Update status
            status.current_phase = new_phase
            status.last_update = current_time
            
            if progress_percentage is not None:
                status.progress_percentage = max(0.0, min(100.0, progress_percentage))
            
            # Update phase completion flags
            if new_phase == ProcessingPhase.NUTRITION_PROCESSING:
                status.vision_completed = True
                status.progress_percentage = max(status.progress_percentage, 33.0)
            elif new_phase == ProcessingPhase.CUISINE_PROCESSING:
                status.nutrition_completed = True
                status.progress_percentage = max(status.progress_percentage, 66.0)
            elif new_phase == ProcessingPhase.FINALIZATION:
                status.cuisine_completed = True
                status.progress_percentage = max(status.progress_percentage, 90.0)
            elif new_phase == ProcessingPhase.COMPLETED:
                status.vision_completed = True
                status.nutrition_completed = True
                status.cuisine_completed = True
                status.progress_percentage = 100.0
            
            self._log_status_change(
                image_id,
                f"Phase updated from {old_phase.value} to {new_phase.value}",
                {
                    "old_phase": old_phase.value,
                    "new_phase": new_phase.value,
                    "progress": status.progress_percentage
                }
            )
            
            self._notify_status_change(image_id, status)
            
            logger.info(f"Phase updated for {image_id}: {old_phase.value} -> {new_phase.value}")
            return status
    
    def report_error(self, image_id: str, error_message: str, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    error_details: Optional[Dict] = None,
                    should_continue: bool = True) -> Optional[ProcessingStatus]:
        """
        Report an error during processing.
        
        Args:
            image_id: Image identifier
            error_message: Error message
            severity: Error severity level
            error_details: Optional detailed error information
            should_continue: Whether processing should continue after this error
            
        Returns:
            Updated ProcessingStatus or None if image_id not found
        """
        with self._lock:
            if image_id not in self.status_store:
                logger.warning(f"Attempted to report error for unknown image {image_id}")
                return None
            
            status = self.status_store[image_id]
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Update error tracking
            status.is_error = True
            status.error_count += 1
            status.last_error = error_message
            status.last_update = current_time
            
            # Add error details
            error_entry = {
                "timestamp": current_time,
                "message": error_message,
                "severity": severity.value,
                "phase": status.current_phase.value,
                "details": error_details or {}
            }
            status.error_details.append(error_entry)
            
            # Determine if processing should stop
            if not should_continue or severity == ErrorSeverity.CRITICAL:
                status.current_phase = ProcessingPhase.FAILED
                status.in_progress = False
                
                # Calculate total execution time
                if image_id in self.phase_start_times:
                    start_time = min(self.phase_start_times[image_id].values())
                    status.total_execution_time = asyncio.get_event_loop().time() - start_time
                
                status.completion_time = current_time
            
            self._log_status_change(
                image_id,
                f"Error reported: {error_message} (severity: {severity.value})",
                {
                    "error_message": error_message,
                    "severity": severity.value,
                    "should_continue": should_continue,
                    "error_count": status.error_count
                }
            )
            
            self._notify_status_change(image_id, status)
            
            logger.error(f"Error reported for {image_id}: {error_message} (severity: {severity.value})")
            return status
    
    def mark_model_completed(self, image_id: str, model_name: str, 
                           success: bool = True, result_data: Optional[Dict] = None) -> Optional[ProcessingStatus]:
        """
        Mark a specific model as completed.
        
        Args:
            image_id: Image identifier
            model_name: Name of the model ('vision', 'nutrition', 'cuisine')
            success: Whether the model completed successfully
            result_data: Optional result data from the model
            
        Returns:
            Updated ProcessingStatus or None if image_id not found
        """
        with self._lock:
            if image_id not in self.status_store:
                logger.warning(f"Attempted to mark model completed for unknown image {image_id}")
                return None
            
            status = self.status_store[image_id]
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Update model completion flags
            if model_name.lower() == 'vision':
                status.vision_completed = success
                if success:
                    status.progress_percentage = max(status.progress_percentage, 33.0)
            elif model_name.lower() == 'nutrition':
                status.nutrition_completed = success
                if success:
                    status.progress_percentage = max(status.progress_percentage, 66.0)
            elif model_name.lower() == 'cuisine':
                status.cuisine_completed = success
                if success:
                    status.progress_percentage = max(status.progress_percentage, 90.0)
            
            status.last_update = current_time
            
            # Store result data in metadata
            if result_data:
                if 'model_results' not in status.metadata:
                    status.metadata['model_results'] = {}
                status.metadata['model_results'][model_name] = {
                    'success': success,
                    'completed_at': current_time,
                    'data_summary': self._summarize_result_data(result_data)
                }
            
            self._log_status_change(
                image_id,
                f"Model {model_name} {'completed successfully' if success else 'failed'}",
                {
                    "model": model_name,
                    "success": success,
                    "progress": status.progress_percentage
                }
            )
            
            self._notify_status_change(image_id, status)
            
            logger.info(f"Model {model_name} marked as {'completed' if success else 'failed'} for {image_id}")
            return status
    
    def finalize_processing(self, image_id: str, overall_success: bool = True,
                          final_result: Optional[Dict] = None) -> Optional[ProcessingStatus]:
        """
        Finalize processing and set final status.
        
        Args:
            image_id: Image identifier
            overall_success: Whether overall processing succeeded
            final_result: Optional final result data
            
        Returns:
            Final ProcessingStatus or None if image_id not found
        """
        with self._lock:
            if image_id not in self.status_store:
                logger.warning(f"Attempted to finalize processing for unknown image {image_id}")
                return None
            
            status = self.status_store[image_id]
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Set final status
            status.in_progress = False
            status.completion_time = current_time
            status.last_update = current_time
            
            if overall_success and not status.is_error:
                status.current_phase = ProcessingPhase.COMPLETED
                status.progress_percentage = 100.0
            else:
                status.current_phase = ProcessingPhase.FAILED
            
            # Calculate total execution time
            if image_id in self.phase_start_times:
                start_time = min(self.phase_start_times[image_id].values())
                status.total_execution_time = asyncio.get_event_loop().time() - start_time
                
                # Record final phase time
                current_phase_start = self.phase_start_times[image_id].get(
                    status.current_phase.value, start_time
                )
                status.phase_execution_times[status.current_phase.value] = (
                    asyncio.get_event_loop().time() - current_phase_start
                )
            
            # Store final result summary
            if final_result:
                status.metadata['final_result_summary'] = self._summarize_result_data(final_result)
            
            # Add completion statistics
            status.metadata['completion_stats'] = {
                'models_completed': sum([
                    status.vision_completed,
                    status.nutrition_completed,
                    status.cuisine_completed
                ]),
                'total_errors': status.error_count,
                'overall_success': overall_success,
                'execution_time_seconds': status.total_execution_time
            }
            
            self._log_status_change(
                image_id,
                f"Processing finalized: {'SUCCESS' if overall_success else 'FAILED'}",
                {
                    "overall_success": overall_success,
                    "total_time": status.total_execution_time,
                    "error_count": status.error_count,
                    "models_completed": status.metadata['completion_stats']['models_completed']
                }
            )
            
            self._notify_status_change(image_id, status)
            
            logger.info(f"Processing finalized for {image_id}: {'SUCCESS' if overall_success else 'FAILED'}")
            return status
    
    def get_status(self, image_id: str) -> Optional[ProcessingStatus]:
        """
        Get current processing status.
        
        Args:
            image_id: Image identifier
            
        Returns:
            ProcessingStatus or None if not found
        """
        with self._lock:
            return self.status_store.get(image_id)
    
    def get_all_active_processing(self) -> Dict[str, ProcessingStatus]:
        """
        Get all currently active processing sessions.
        
        Returns:
            Dictionary of image_id -> ProcessingStatus for active sessions
        """
        with self._lock:
            return {
                image_id: status 
                for image_id, status in self.status_store.items()
                if status.in_progress
            }
    
    def get_processing_summary(self, image_id: str) -> Optional[Dict]:
        """
        Get a comprehensive summary of processing status.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Dictionary with processing summary or None if not found
        """
        with self._lock:
            status = self.status_store.get(image_id)
            if not status:
                return None
            
            return {
                "image_id": image_id,
                "status": {
                    "in_progress": status.in_progress,
                    "is_error": status.is_error,
                    "current_phase": status.current_phase.value,
                    "progress_percentage": status.progress_percentage
                },
                "timing": {
                    "start_time": status.start_time,
                    "last_update": status.last_update,
                    "completion_time": status.completion_time,
                    "total_execution_time": status.total_execution_time,
                    "phase_times": status.phase_execution_times
                },
                "models": {
                    "vision_completed": status.vision_completed,
                    "nutrition_completed": status.nutrition_completed,
                    "cuisine_completed": status.cuisine_completed
                },
                "errors": {
                    "error_count": status.error_count,
                    "last_error": status.last_error,
                    "error_details": status.error_details[-3:] if status.error_details else []  # Last 3 errors
                },
                "metadata": status.metadata
            }
    
    def _summarize_result_data(self, data: Dict) -> Dict:
        """
        Create a summary of result data for storage.
        
        Args:
            data: Result data to summarize
            
        Returns:
            Summarized data
        """
        summary = {}
        
        # Common fields to extract
        if isinstance(data, dict):
            for key in ['confidence', 'status', 'analysis_status', 'error_message']:
                if key in data:
                    summary[key] = data[key]
            
            # Count items in lists
            for key, value in data.items():
                if isinstance(value, list):
                    summary[f"{key}_count"] = len(value)
                elif isinstance(value, dict):
                    summary[f"{key}_keys"] = list(value.keys())[:5]  # First 5 keys
        
        return summary
    
    def cleanup_old_status(self, max_age_hours: int = 24):
        """
        Clean up old status entries to free memory.
        
        Args:
            max_age_hours: Maximum age in hours for keeping status entries
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        with self._lock:
            to_remove = []
            
            for image_id, status in self.status_store.items():
                # Parse start time
                try:
                    start_time = datetime.fromisoformat(status.start_time.replace('Z', '+00:00'))
                    if start_time.timestamp() < cutoff_time and not status.in_progress:
                        to_remove.append(image_id)
                except ValueError:
                    # If we can't parse the time, remove it
                    if not status.in_progress:
                        to_remove.append(image_id)
            
            # Remove old entries
            for image_id in to_remove:
                del self.status_store[image_id]
                if image_id in self.phase_start_times:
                    del self.phase_start_times[image_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old status entries")


# Convenience functions for common operations
def initialize_processing_status(image_id: str, status_manager: StatusManager, 
                               metadata: Optional[Dict] = None) -> ProcessingStatus:
    """Initialize processing status for an image."""
    return status_manager.initialize_processing(image_id, metadata)


def update_processing_phase(image_id: str, status_manager: StatusManager,
                          phase: ProcessingPhase, progress: Optional[float] = None) -> Optional[ProcessingStatus]:
    """Update processing phase for an image."""
    return status_manager.update_phase(image_id, phase, progress)


def finalize_processing_status(image_id: str, status_manager: StatusManager,
                             success: bool = True, result: Optional[Dict] = None) -> Optional[ProcessingStatus]:
    """Finalize processing status for an image."""
    return status_manager.finalize_processing(image_id, success, result)