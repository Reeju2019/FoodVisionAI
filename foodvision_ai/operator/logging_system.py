"""
Model_Remark Logging System for FoodVisionAI

Implements comprehensive logging for all model executions and function calls,
JSON list management for execution tracking, and debugging utilities.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import os
from loguru import logger


class LogLevel(Enum):
    """Log levels for model remarks."""
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Component types for logging."""
    OPERATOR = "operator"
    VISION_MODEL = "vision_model"
    NUTRITION_LLM = "nutrition_llm"
    CUISINE_CLASSIFIER = "cuisine_classifier"
    DATABASE = "database"
    API = "api"
    SYSTEM = "system"


@dataclass
class ModelRemark:
    """Single model remark entry with comprehensive metadata."""
    id: str
    timestamp: str
    component: ComponentType
    level: LogLevel
    message: str
    function_name: Optional[str] = None
    execution_time_ms: Optional[float] = None
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    error_details: Optional[Dict] = None
    metadata: Optional[Dict] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enums to strings
        result['component'] = self.component.value
        result['level'] = self.level.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelRemark':
        """Create ModelRemark from dictionary."""
        # Convert string values back to enums
        data['component'] = ComponentType(data['component'])
        data['level'] = LogLevel(data['level'])
        return cls(**data)


class ModelRemarkLogger:
    """
    Comprehensive logging system for model executions and function calls.
    
    Manages JSON list storage, provides debugging utilities, and tracks
    all execution messages for comprehensive system monitoring.
    """
    
    def __init__(self, log_file_path: Optional[str] = None, 
                 max_remarks_per_session: int = 1000,
                 enable_file_logging: bool = True):
        """
        Initialize the Model Remark Logger.
        
        Args:
            log_file_path: Path to log file (defaults to logs/model_remarks.json)
            max_remarks_per_session: Maximum remarks to keep in memory per session
            enable_file_logging: Whether to enable file-based logging
        """
        self.max_remarks_per_session = max_remarks_per_session
        self.enable_file_logging = enable_file_logging
        
        # Set up log file path
        if log_file_path is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self.log_file_path = log_dir / "model_remarks.json"
        else:
            self.log_file_path = Path(log_file_path)
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for active sessions
        self.session_remarks: Dict[str, List[ModelRemark]] = {}
        self.global_remarks: List[ModelRemark] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Session tracking
        self.active_sessions: Dict[str, Dict] = {}
        
        logger.info(f"Model Remark Logger initialized with file: {self.log_file_path}")
    
    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> str:
        """
        Create a new logging session.
        
        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        with self._lock:
            if session_id in self.session_remarks:
                logger.warning(f"Session {session_id} already exists, clearing previous remarks")
            
            self.session_remarks[session_id] = []
            self.active_sessions[session_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
                "remark_count": 0,
                "last_activity": datetime.now(timezone.utc).isoformat()
            }
            
            # Log session creation
            self._add_remark_internal(
                session_id=session_id,
                component=ComponentType.SYSTEM,
                level=LogLevel.INFO,
                message=f"Logging session created: {session_id}",
                metadata=metadata
            )
            
            return session_id
    
    def _generate_remark_id(self) -> str:
        """Generate unique remark ID."""
        return str(uuid.uuid4())
    
    def _add_remark_internal(self, session_id: str, component: ComponentType, 
                           level: LogLevel, message: str, **kwargs) -> ModelRemark:
        """
        Internal method to add a remark with full control.
        
        Args:
            session_id: Session identifier
            component: Component that generated the remark
            level: Log level
            message: Log message
            **kwargs: Additional remark fields
            
        Returns:
            Created ModelRemark
        """
        remark = ModelRemark(
            id=self._generate_remark_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            component=component,
            level=level,
            message=message,
            **kwargs
        )
        
        with self._lock:
            # Add to session remarks
            if session_id not in self.session_remarks:
                self.session_remarks[session_id] = []
            
            self.session_remarks[session_id].append(remark)
            
            # Maintain session size limit
            if len(self.session_remarks[session_id]) > self.max_remarks_per_session:
                removed = self.session_remarks[session_id].pop(0)
                logger.debug(f"Removed oldest remark from session {session_id}: {removed.id}")
            
            # Add to global remarks
            self.global_remarks.append(remark)
            
            # Update session tracking
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["remark_count"] += 1
                self.active_sessions[session_id]["last_activity"] = remark.timestamp
            
            # File logging
            if self.enable_file_logging:
                self._write_to_file(remark)
        
        return remark
    
    def log_function_call(self, session_id: str, component: ComponentType, 
                         function_name: str, input_data: Optional[Dict] = None,
                         correlation_id: Optional[str] = None) -> str:
        """
        Log the start of a function call.
        
        Args:
            session_id: Session identifier
            component: Component making the call
            function_name: Name of the function being called
            input_data: Input parameters (will be sanitized)
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            Correlation ID for tracking this function call
        """
        if correlation_id is None:
            correlation_id = self._generate_remark_id()
        
        # Sanitize input data (remove sensitive information)
        sanitized_input = self._sanitize_data(input_data) if input_data else None
        
        self._add_remark_internal(
            session_id=session_id,
            component=component,
            level=LogLevel.INFO,
            message=f"Function call started: {function_name}",
            function_name=function_name,
            input_data=sanitized_input,
            correlation_id=correlation_id,
            metadata={"call_type": "start"}
        )
        
        return correlation_id
    
    def log_function_result(self, session_id: str, component: ComponentType,
                          function_name: str, output_data: Optional[Dict] = None,
                          execution_time_ms: Optional[float] = None,
                          correlation_id: Optional[str] = None,
                          success: bool = True) -> ModelRemark:
        """
        Log the result of a function call.
        
        Args:
            session_id: Session identifier
            component: Component that completed the call
            function_name: Name of the function that completed
            output_data: Output data (will be sanitized)
            execution_time_ms: Execution time in milliseconds
            correlation_id: Correlation ID from the start call
            success: Whether the function succeeded
            
        Returns:
            Created ModelRemark
        """
        # Sanitize output data
        sanitized_output = self._sanitize_data(output_data) if output_data else None
        
        level = LogLevel.SUCCESS if success else LogLevel.ERROR
        message = f"Function call {'completed' if success else 'failed'}: {function_name}"
        
        if execution_time_ms is not None:
            message += f" (took {execution_time_ms:.2f}ms)"
        
        return self._add_remark_internal(
            session_id=session_id,
            component=component,
            level=level,
            message=message,
            function_name=function_name,
            output_data=sanitized_output,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id,
            metadata={"call_type": "result", "success": success}
        )
    
    def log_error(self, session_id: str, component: ComponentType,
                  error_message: str, error_details: Optional[Dict] = None,
                  function_name: Optional[str] = None,
                  correlation_id: Optional[str] = None) -> ModelRemark:
        """
        Log an error with detailed information.
        
        Args:
            session_id: Session identifier
            component: Component that encountered the error
            error_message: Error message
            error_details: Detailed error information
            function_name: Function where error occurred
            correlation_id: Optional correlation ID
            
        Returns:
            Created ModelRemark
        """
        return self._add_remark_internal(
            session_id=session_id,
            component=component,
            level=LogLevel.ERROR,
            message=error_message,
            function_name=function_name,
            error_details=error_details,
            correlation_id=correlation_id,
            metadata={"error_logged": True}
        )
    
    def log_info(self, session_id: str, component: ComponentType,
                 message: str, metadata: Optional[Dict] = None,
                 correlation_id: Optional[str] = None) -> ModelRemark:
        """
        Log an informational message.
        
        Args:
            session_id: Session identifier
            component: Component logging the message
            message: Information message
            metadata: Optional metadata
            correlation_id: Optional correlation ID
            
        Returns:
            Created ModelRemark
        """
        return self._add_remark_internal(
            session_id=session_id,
            component=component,
            level=LogLevel.INFO,
            message=message,
            metadata=metadata,
            correlation_id=correlation_id
        )
    
    def log_warning(self, session_id: str, component: ComponentType,
                   message: str, metadata: Optional[Dict] = None,
                   correlation_id: Optional[str] = None) -> ModelRemark:
        """
        Log a warning message.
        
        Args:
            session_id: Session identifier
            component: Component logging the warning
            message: Warning message
            metadata: Optional metadata
            correlation_id: Optional correlation ID
            
        Returns:
            Created ModelRemark
        """
        return self._add_remark_internal(
            session_id=session_id,
            component=component,
            level=LogLevel.WARNING,
            message=message,
            metadata=metadata,
            correlation_id=correlation_id
        )
    
    def _sanitize_data(self, data: Any) -> Any:
        """
        Sanitize data to remove sensitive information and limit size.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if data is None:
            return None
        
        # Convert to JSON string and back to ensure serializability
        try:
            json_str = json.dumps(data, default=str)
            
            # Limit size to prevent huge logs
            max_size = 10000  # 10KB limit
            if len(json_str) > max_size:
                return {"_truncated": True, "_size": len(json_str), "_preview": json_str[:1000]}
            
            return json.loads(json_str)
        except (TypeError, ValueError) as e:
            return {"_serialization_error": str(e), "_type": str(type(data))}
    
    def _write_to_file(self, remark: ModelRemark):
        """
        Write a remark to the log file.
        
        Args:
            remark: ModelRemark to write
        """
        try:
            # Append to file as JSON lines
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                json.dump(remark.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write remark to file: {e}")
    
    def get_session_remarks(self, session_id: str, 
                          level_filter: Optional[LogLevel] = None,
                          component_filter: Optional[ComponentType] = None,
                          limit: Optional[int] = None) -> List[ModelRemark]:
        """
        Get remarks for a specific session with optional filtering.
        
        Args:
            session_id: Session identifier
            level_filter: Optional log level filter
            component_filter: Optional component filter
            limit: Optional limit on number of remarks
            
        Returns:
            List of filtered ModelRemarks
        """
        with self._lock:
            remarks = self.session_remarks.get(session_id, [])
            
            # Apply filters
            if level_filter is not None:
                remarks = [r for r in remarks if r.level == level_filter]
            
            if component_filter is not None:
                remarks = [r for r in remarks if r.component == component_filter]
            
            # Apply limit
            if limit is not None:
                remarks = remarks[-limit:]  # Get most recent
            
            return remarks.copy()
    
    def get_session_summary(self, session_id: str) -> Dict:
        """
        Get a summary of a session's logging activity.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session summary
        """
        with self._lock:
            remarks = self.session_remarks.get(session_id, [])
            session_info = self.active_sessions.get(session_id, {})
            
            # Count by level
            level_counts = {}
            for level in LogLevel:
                level_counts[level.value] = sum(1 for r in remarks if r.level == level)
            
            # Count by component
            component_counts = {}
            for component in ComponentType:
                component_counts[component.value] = sum(1 for r in remarks if r.component == component)
            
            # Get error details
            errors = [r for r in remarks if r.level == LogLevel.ERROR]
            
            return {
                "session_id": session_id,
                "session_info": session_info,
                "total_remarks": len(remarks),
                "level_counts": level_counts,
                "component_counts": component_counts,
                "error_count": len(errors),
                "recent_errors": [r.to_dict() for r in errors[-5:]],  # Last 5 errors
                "first_remark": remarks[0].timestamp if remarks else None,
                "last_remark": remarks[-1].timestamp if remarks else None
            }
    
    def export_session_logs(self, session_id: str, output_file: str) -> bool:
        """
        Export session logs to a file.
        
        Args:
            session_id: Session identifier
            output_file: Output file path
            
        Returns:
            True if export succeeded, False otherwise
        """
        try:
            remarks = self.get_session_remarks(session_id)
            
            export_data = {
                "session_id": session_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "summary": self.get_session_summary(session_id),
                "remarks": [r.to_dict() for r in remarks]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(remarks)} remarks for session {session_id} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export session logs: {e}")
            return False
    
    def close_session(self, session_id: str) -> Dict:
        """
        Close a logging session and return summary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dictionary
        """
        with self._lock:
            # Get final summary
            summary = self.get_session_summary(session_id)
            
            # Log session closure
            self._add_remark_internal(
                session_id=session_id,
                component=ComponentType.SYSTEM,
                level=LogLevel.INFO,
                message=f"Logging session closed: {session_id}",
                metadata={"session_summary": summary}
            )
            
            # Mark session as closed
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["closed_at"] = datetime.now(timezone.utc).isoformat()
                self.active_sessions[session_id]["status"] = "closed"
            
            return summary
    
    def cleanup_old_sessions(self, max_age_hours: int = 24, keep_closed: bool = False):
        """
        Clean up old sessions to free memory.
        
        Args:
            max_age_hours: Maximum age in hours for keeping sessions
            keep_closed: Whether to keep closed sessions longer
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        with self._lock:
            sessions_to_remove = []
            
            for session_id, session_info in self.active_sessions.items():
                created_at = datetime.fromisoformat(session_info["created_at"].replace('Z', '+00:00'))
                
                # Check if session is old enough to remove
                if created_at.timestamp() < cutoff_time:
                    # If keep_closed is True, only remove if not closed or very old
                    if not keep_closed or session_info.get("status") != "closed":
                        sessions_to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in sessions_to_remove:
                if session_id in self.session_remarks:
                    del self.session_remarks[session_id]
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
            
            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} old logging sessions")


# Global logger instance
_global_logger: Optional[ModelRemarkLogger] = None


def get_global_logger() -> ModelRemarkLogger:
    """Get or create the global model remark logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ModelRemarkLogger()
    return _global_logger


def set_global_logger(logger_instance: ModelRemarkLogger):
    """Set the global model remark logger."""
    global _global_logger
    _global_logger = logger_instance


# Convenience functions for common logging operations
def create_logging_session(session_id: str, metadata: Optional[Dict] = None) -> str:
    """Create a new logging session using the global logger."""
    return get_global_logger().create_session(session_id, metadata)


def log_function_call(session_id: str, component: ComponentType, 
                     function_name: str, input_data: Optional[Dict] = None) -> str:
    """Log a function call using the global logger."""
    return get_global_logger().log_function_call(session_id, component, function_name, input_data)


def log_function_result(session_id: str, component: ComponentType,
                       function_name: str, output_data: Optional[Dict] = None,
                       execution_time_ms: Optional[float] = None,
                       correlation_id: Optional[str] = None,
                       success: bool = True) -> ModelRemark:
    """Log a function result using the global logger."""
    return get_global_logger().log_function_result(
        session_id, component, function_name, output_data, 
        execution_time_ms, correlation_id, success
    )


def log_error(session_id: str, component: ComponentType,
              error_message: str, error_details: Optional[Dict] = None,
              function_name: Optional[str] = None) -> ModelRemark:
    """Log an error using the global logger."""
    return get_global_logger().log_error(session_id, component, error_message, error_details, function_name)


def log_info(session_id: str, component: ComponentType,
             message: str, metadata: Optional[Dict] = None) -> ModelRemark:
    """Log an info message using the global logger."""
    return get_global_logger().log_info(session_id, component, message, metadata)


def get_session_summary(session_id: str) -> Dict:
    """Get session summary using the global logger."""
    return get_global_logger().get_session_summary(session_id)