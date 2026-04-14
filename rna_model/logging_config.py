"""Logging configuration for RNA 3D folding pipeline."""

import logging
import sys
import json
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    structured: bool = False
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured: Whether to use structured logging
        
    Returns:
        Configured logger instance
    """
    if structured:
        try:
            from .structured_logger import StructuredLogger
            return StructuredLogger(name, log_file, level)
        except ImportError:
            # Fallback to standard logging if structured logger not available
            structured = False
    
    # Use standard logging when structured logging not available
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove our handlers to avoid duplicates, but keep system handlers
    handlers_to_remove = [h for h in logger.handlers if hasattr(h, '_rna_logger')]
    for handler in handlers_to_remove:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler._rna_logger = True  # Mark as our handler
    
    # Console formatter
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler._rna_logger = True  # Mark as our handler
        
        # File formatter with more details
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logging(name)


class StructuredLogger:
    """Structured logger with JSON output for better log parsing."""
    
    def __init__(self, name: str, log_file: Optional[Union[str, Path]], level: str):
        # Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Logger name must be a non-empty string")
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
        
        self.name = name
        self.log_file = Path(log_file) if log_file else None
        self.level = getattr(logging, level.upper())
        self.logger = logging.getLogger(f"{name}_structured")
        self.logger.setLevel(self.level)
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers for structured logging."""
        
        # Remove our handlers to avoid duplicates, but keep system handlers
        handlers_to_remove = [h for h in self.logger.handlers if hasattr(h, '_rna_structured_logger')]
        for handler in handlers_to_remove:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler._rna_structured_logger = True  # Mark as our handler
        console_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.level)
            file_handler._rna_structured_logger = True  # Mark as our handler
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, extra={'structured_data': kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, extra={'structured_data': kwargs})
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, extra={'structured_data': kwargs})
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, extra={'structured_data': kwargs})
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self.logger.critical(message, extra={'structured_data': kwargs})


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add structured data if available
        if hasattr(record, 'structured_data') and record.structured_data:
            log_entry.update(record.structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)