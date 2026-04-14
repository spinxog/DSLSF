if structured:
        return StructuredLogger(name, log_file, level)
    else:
        # Standard logging is acceptable as a fallback for logging infrastructure
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))