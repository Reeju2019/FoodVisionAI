"""
Operator Layer for FoodVisionAI

Orchestrates model execution, handles data synchronization between models
and database, and manages error handling and logging.
"""

from .core import OperatorCore, ProcessingResult, ModelResult, ModelStage, ProcessingStatus
from .logging_system import (
    ModelRemarkLogger, 
    ModelRemark, 
    ComponentType, 
    LogLevel,
    create_logging_session,
    log_function_call,
    log_function_result,
    log_error,
    log_info,
    get_session_summary
)
from .status_manager import (
    StatusManager, 
    ProcessingPhase, 
    ErrorSeverity,
    ProcessingStatus as StatusManagerProcessingStatus,
    initialize_processing_status,
    update_processing_phase,
    finalize_processing_status
)
from .pipeline_integration import (
    IntegratedPipeline,
    test_pipeline_with_sample_images,
    test_error_scenarios,
    run_complete_pipeline_test
)
from .database_integration import (
    DatabaseIntegratedOperator,
    trigger_analysis_pipeline
)

__all__ = [
    # Core components
    'OperatorCore',
    'ProcessingResult',
    'ModelResult', 
    'ModelStage',
    'ProcessingStatus',
    
    # Logging system
    'ModelRemarkLogger',
    'ModelRemark',
    'ComponentType',
    'LogLevel',
    'create_logging_session',
    'log_function_call',
    'log_function_result',
    'log_error',
    'log_info',
    'get_session_summary',
    
    # Status management
    'StatusManager',
    'ProcessingPhase',
    'ErrorSeverity',
    'StatusManagerProcessingStatus',
    'initialize_processing_status',
    'update_processing_phase',
    'finalize_processing_status',
    
    # Integrated pipeline
    'IntegratedPipeline',
    'test_pipeline_with_sample_images',
    'test_error_scenarios',
    'run_complete_pipeline_test',
    
    # Database integration
    'DatabaseIntegratedOperator',
    'trigger_analysis_pipeline'
]