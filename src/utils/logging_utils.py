# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "Logging Utilities"
# script_id: "fr_11_uc_091_ec_09_tc_253"
# gps_coordinate: "fr_11_uc_091_ec_09_tc_253"
# script_name: "src/utils/logging_utils.py"
# purpose: "Standardized logging utilities for deployment with no-op shims"
# =============================================================================

import logging
import sys
from typing import Optional, Any, Dict

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def setup_basic_logging(level: str = "INFO", **kwargs) -> logging.Logger:
    """Setup basic logging configuration."""
    # Idempotent basicConfig guard
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            stream=sys.stdout
        )
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    
    return logging.getLogger()

def log_prompt_response(prompt=None, response=None, meta=None, **kwargs) -> bool:
    """Log prompt and response interaction (no-op shim for compatibility)."""
    logger = get_logger("prompt_logger")
    
    # Optional logging for debugging
    if prompt or response:
        logger.debug(f"Prompt-Response logged: prompt_len={len(str(prompt)) if prompt else 0}, "
                    f"response_len={len(str(response)) if response else 0}")
    
    return True

def log_event(event_name=None, payload=None, level="info", **kwargs) -> bool:
    """Log application events (no-op shim for compatibility)."""
    logger = get_logger("event_logger")
    
    # Get the appropriate log method
    log_method = getattr(logger, level.lower(), logger.info)
    
    # Optional logging for debugging
    if event_name:
        log_method(f"Event logged: {event_name} (payload_type={type(payload).__name__})")
    
    return True

def redact_sensitive_info(text: str, redaction_char: str = "*") -> str:
    """Redact sensitive information from log messages."""
    if not text or len(text) < 4:
        return redaction_char * 8
    
    if len(text) <= 8:
        return text[:2] + redaction_char * 4 + text[-2:]
    
    return text[:2] + redaction_char * (len(text) - 4) + text[-2:]

# Additional compatibility shims for common logging patterns
def log_request(request_data=None, **kwargs) -> bool:
    """Log HTTP request data (no-op shim)."""
    return log_event("http_request", request_data, level="info", **kwargs)

def log_response(response_data=None, **kwargs) -> bool:
    """Log HTTP response data (no-op shim).""" 
    return log_event("http_response", response_data, level="info", **kwargs)

def log_error(error_msg=None, exception=None, **kwargs) -> bool:
    """Log error information (no-op shim)."""
    return log_event("error", {"message": error_msg, "exception": str(exception)}, level="error", **kwargs)
def log_prompt_response(prompt=None, response=None, meta=None, **kwargs) -> bool:
    """Log prompt and response interaction (no-op shim for compatibility)."""
    logger = get_logger("prompt_logger")
    
    # Optional logging for debugging
    if prompt or response:
        logger.debug(f"Prompt-Response logged: prompt_len={len(str(prompt)) if prompt else 0}, "
                    f"response_len={len(str(response)) if response else 0}")
    
    return True

def log_event(event_name=None, payload=None, level="info", **kwargs) -> bool:
    """Log application events (no-op shim for compatibility)."""
    logger = get_logger("event_logger")
    
    # Get the appropriate log method
    log_method = getattr(logger, level.lower(), logger.info)
    
    # Optional logging for debugging
    if event_name:
        log_method(f"Event logged: {event_name} (payload_type={type(payload).__name__})")
    
    return True

def log_request(request_data=None, **kwargs) -> bool:
    """Log HTTP request data (no-op shim)."""
    return log_event("http_request", request_data, level="info", **kwargs)

def log_response(response_data=None, **kwargs) -> bool:
    """Log HTTP response data (no-op shim).""" 
    return log_event("http_response", response_data, level="info", **kwargs)

def log_error(error_msg=None, exception=None, **kwargs) -> bool:
    """Log error information (no-op shim)."""
    return log_event("error", {"message": error_msg, "exception": str(exception)}, level="error", **kwargs)    