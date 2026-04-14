"""Centralized error handling and logging utilities for RNA 3D folding pipeline.

This module provides standardized error messages, logging utilities,
and exception classes to ensure consistent error reporting throughout the codebase.
"""

import logging
import sys
import traceback
from typing import Optional, Dict, Any, Union
from enum import Enum
from pathlib import Path

from .constants import LOGGING


class ErrorCategory(Enum):
    """Categories of errors for better organization and handling."""
    
    VALIDATION = "validation"
    IO_ERROR = "io_error"
    MEMORY_ERROR = "memory_error"
    COMPUTATION_ERROR = "computation_error"
    CONFIGURATION_ERROR = "configuration_error"
    SECURITY_ERROR = "security_error"
    TRAINING_ERROR = "training_error"
    GEOMETRY_ERROR = "geometry_error"
    DATA_ERROR = "data_error"


class RNAError(Exception):
    """Base exception class for RNA 3D folding pipeline."""
    
    def __init__(self, message: str, category: ErrorCategory = None, 
                 context: Dict[str, Any] = None, cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.category = category or ErrorCategory.COMPUTATION_ERROR
        self.context = context or {}
        self.cause = cause
    
    def __str__(self) -> str:
        base_msg = f"[{self.category.value.upper()}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        if self.cause:
            base_msg += f" (Caused by: {type(self.cause).__name__}: {self.cause})"
        return base_msg


class ValidationError(RNAError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)


class IOError(RNAError):
    """Raised when file I/O operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.IO_ERROR, **kwargs)


class MemoryError(RNAError):
    """Raised when memory operations fail or limits are exceeded."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.MEMORY_ERROR, **kwargs)


class ComputationError(RNAError):
    """Raised when computational operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.COMPUTATION_ERROR, **kwargs)


class ConfigurationError(RNAError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.CONFIGURATION_ERROR, **kwargs)


class SecurityError(RNAError):
    """Raised when security violations are detected."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.SECURITY_ERROR, **kwargs)


class TrainingError(RNAError):
    """Raised when training operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.TRAINING_ERROR, **kwargs)


class GeometryError(RNAError):
    """Raised when geometric computations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.GEOMETRY_ERROR, **kwargs)


class DataError(RNAError):
    """Raised when data operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.DATA_ERROR, **kwargs)


class LoggerManager:
    """Manages logging configuration and provides standardized logging utilities."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def setup_logging(cls, log_level: str = None, log_file: Optional[Union[str, Path]] = None,
                     log_format: str = None) -> None:
        """Setup logging configuration for the entire pipeline."""
        if cls._configured:
            return
        
        # Use defaults from constants if not provided
        log_level = log_level or LOGGING.DEFAULT_LOG_LEVEL
        log_format = log_format or LOGGING.LOG_FORMAT
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[]
        )
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        if not cls._configured:
            cls.setup_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def log_exception(cls, logger: logging.Logger, exception: Exception, 
                     context: Dict[str, Any] = None) -> None:
        """Log an exception with full context and traceback."""
        error_msg = str(exception)
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            error_msg += f" | Context: {context_str}"
        
        logger.error(error_msg, exc_info=True)
        
        # Log the full traceback
        logger.debug(f"Full traceback for {type(exception).__name__}:\n" + 
                    "".join(traceback.format_exception(type(exception), exception, exception.__traceback__)))


class ErrorHandler:
    """Standardized error handling utilities."""
    
    @staticmethod
    def handle_validation_error(value: Any, expected_type: type, field_name: str, 
                                additional_info: str = None) -> ValidationError:
        """Create a standardized validation error."""
        message = f"Invalid {field_name}: expected {expected_type.__name__}, got {type(value).__name__}"
        if additional_info:
            message += f". {additional_info}"
        
        return ValidationError(message, context={
            'field_name': field_name,
            'expected_type': expected_type.__name__,
            'actual_type': type(value).__name__,
            'actual_value': str(value)
        })
    
    @staticmethod
    def handle_range_error(value: Union[int, float], min_val: Union[int, float], 
                          max_val: Union[int, float], field_name: str) -> ValidationError:
        """Create a standardized range validation error."""
        return ValidationError(
            f"{field_name} must be between {min_val} and {max_val}, got {value}",
            context={
                'field_name': field_name,
                'value': value,
                'min_value': min_val,
                'max_value': max_val
            }
        )
    
    @staticmethod
    def handle_file_error(file_path: Union[str, Path], operation: str, 
                         original_error: Exception = None) -> IOError:
        """Create a standardized file I/O error."""
        message = f"Failed to {operation} file: {file_path}"
        if original_error:
            message += f" ({original_error})"
        
        return IOError(message, context={
            'file_path': str(file_path),
            'operation': operation
        }, cause=original_error)
    
    @staticmethod
    def handle_memory_error(operation: str, size_mb: float = None, 
                          available_mb: float = None) -> MemoryError:
        """Create a standardized memory error."""
        message = f"Memory error during {operation}"
        if size_mb is not None:
            message += f" (requested: {size_mb:.1f} MB)"
        if available_mb is not None:
            message += f" (available: {available_mb:.1f} MB)"
        
        return MemoryError(message, context={
            'operation': operation,
            'requested_size_mb': size_mb,
            'available_size_mb': available_mb
        })
    
    @staticmethod
    def handle_computation_error(operation: str, shape: tuple = None, 
                                original_error: Exception = None) -> ComputationError:
        """Create a standardized computation error."""
        message = f"Computation error in {operation}"
        if shape:
            message += f" with shape {shape}"
        if original_error:
            message += f": {original_error}"
        
        return ComputationError(message, context={
            'operation': operation,
            'shape': shape
        }, cause=original_error)
    
    @staticmethod
    def handle_security_error(operation: str, details: str = None, 
                             file_path: Union[str, Path] = None) -> SecurityError:
        """Create a standardized security error."""
        message = f"Security violation in {operation}"
        if details:
            message += f": {details}"
        if file_path:
            message += f" (file: {file_path})"
        
        return SecurityError(message, context={
            'operation': operation,
            'details': details,
            'file_path': str(file_path) if file_path else None
        })
    
    @staticmethod
    def log_and_raise_error(logger: logging.Logger, error: Exception, 
                           reraise: bool = True) -> None:
        """Log an error and optionally reraise it."""
        LoggerManager.log_exception(logger, error)
        if reraise:
            raise error


# Standardized error message templates
ERROR_MESSAGES = {
    'invalid_type': "Invalid type for {field_name}: expected {expected_type}, got {actual_type}",
    'out_of_range': "{field_name} ({value}) is out of range [{min_val}, {max_val}]",
    'file_not_found': "File not found: {file_path}",
    'permission_denied': "Permission denied: {file_path}",
    'invalid_format': "Invalid format: {details}",
    'memory_exhausted': "Memory exhausted during {operation}",
    'computation_failed': "Computation failed: {operation} ({error})",
    'security_violation': "Security violation: {details}",
    'configuration_invalid': "Invalid configuration: {details}",
    'validation_failed': "Validation failed: {field_name} - {reason}",
    'nan_detected': "NaN values detected in {field_name}",
    'inf_detected': "Infinite values detected in {field_name}",
    'shape_mismatch': "Shape mismatch: expected {expected_shape}, got {actual_shape}",
    'dimension_mismatch': "Dimension mismatch: expected {expected_dim}D, got {actual_dim}D",
}


def format_error_message(template_key: str, **kwargs) -> str:
    """Format an error message using a template."""
    template = ERROR_MESSAGES.get(template_key, "Unknown error: {details}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"Error formatting message: {e}. Template: {template}, Args: {kwargs}"


def safe_execute(operation: str, func, logger: logging.Logger = None, 
                 error_category: ErrorCategory = ErrorCategory.COMPUTATION_ERROR,
                 reraise: bool = True, **kwargs) -> Any:
    """Safely execute a function with standardized error handling."""
    try:
        return func(**kwargs)
    except Exception as e:
        # Create appropriate error based on exception type
        if isinstance(e, (FileNotFoundError, PermissionError, OSError)):
            error = IOError(f"File operation failed in {operation}: {e}", cause=e)
        elif isinstance(e, (MemoryError, RuntimeError)) and "memory" in str(e).lower():
            error = MemoryError(f"Memory error in {operation}: {e}", cause=e)
        elif isinstance(e, (ValueError, TypeError)):
            error = ValidationError(f"Validation error in {operation}: {e}", cause=e)
        else:
            error = RNAError(f"Error in {operation}: {e}", category=error_category, cause=e)
        
        if logger:
            LoggerManager.log_exception(logger, error)
        
        if reraise:
            raise error
        
        return None


# Convenience functions for common operations
def get_logger(name: str) -> logging.Logger:
    """Get a logger with standardized configuration."""
    return LoggerManager.get_logger(name)


def log_error(logger: logging.Logger, message: str, category: ErrorCategory = None, 
              context: Dict[str, Any] = None, exc_info: bool = False) -> None:
    """Log an error with standardized format."""
    if category:
        message = f"[{category.value.upper()}] {message}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" | Context: {context_str}"
    
    logger.error(message, exc_info=exc_info)


def log_warning(logger: logging.Logger, message: str, context: Dict[str, Any] = None) -> None:
    """Log a warning with standardized format."""
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" | Context: {context_str}"
    
    logger.warning(message)


def log_info(logger: logging.Logger, message: str, context: Dict[str, Any] = None) -> None:
    """Log an info message with standardized format."""
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" | Context: {context_str}"
    
    logger.info(message)


# Initialize default logging setup
LoggerManager.setup_logging()