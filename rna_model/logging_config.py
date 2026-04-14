if structured:
        return StructuredLogger(name, log_file, level)
    else:
        # Use standard logging when structured logging not available
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))