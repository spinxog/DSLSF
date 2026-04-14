"""Structured logging configuration for RNA 3D folding pipeline."""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredLogger:
    """Structured logger with JSON output and performance tracking."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (structured JSON)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with structured data."""
        extra = kwargs if kwargs else {}
        self.logger.log(level, message, extra=extra)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add structured data from extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, base_logger: StructuredLogger):
        self.logger = base_logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a named timer."""
        import time
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, **metadata):
        """End a named timer and log the duration."""
        import time
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        self.logger.info(f"Timer '{name}' completed", 
                         duration=duration, 
                         duration_ms=duration * 1000,
                         **metadata)
    
    def log_memory_usage(self, **metadata):
        """Log current memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                memory_info = {
                    'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                    'cached_gb': torch.cuda.memory_reserved() / 1024**3,
                    'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
                }
                self.logger.info("GPU memory usage", **memory_info, **metadata)
        except ImportError:
            pass
    
    def log_model_stats(self, model, **metadata):
        """Log model statistics."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / 1024**2
            }
            
            self.logger.info("Model statistics", **model_info, **metadata)
        except Exception as e:
            self.logger.error(f"Failed to log model stats: {e}")


def setup_logger(name: str, 
                 log_dir: Optional[Path] = None,
                 level: str = "INFO",
                 structured: bool = True) -> StructuredLogger:
    """Setup logger with optional structured logging."""
    
    log_file = None
    if log_dir:
        log_file = log_dir / f"{name}.json"
    
    if structured:
        return StructuredLogger(name, log_file, level)
    else:
        # Fallback to standard logging
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
