import sys
from pathlib import Path
from loguru import logger
from typing import Optional

def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
    log_to_console: bool = True,
    log_to_file: bool = True,
    colorize: bool = True
) -> logger:
    # Remove default handler
    logger.remove()
    
    # Format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Console handler
    if log_to_console:
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=colorize
        )
    
    # File handler
    if log_to_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8"
        )
    
    return logger

def get_logger(name: str) -> logger:
    return logger.bind(name=name)
