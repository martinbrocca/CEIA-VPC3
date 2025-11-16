"""
Logging utilities
Setup and configuration for logging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'training',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and/or console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_experiment_dir(base_dir: str = 'outputs/experiments') -> Path:
    """
    Create timestamped experiment directory
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to created experiment directory
    """
    timestamp = get_timestamp()
    exp_dir = Path(base_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'models').mkdir(exist_ok=True)
    (exp_dir / 'figures').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    
    return exp_dir


def log_gpu_info(logger: logging.Logger):
    """
    Log GPU information
    
    Args:
        logger: Logger instance
    """
    import torch
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            logger.info(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        logger.info("CUDA available: No")


def log_system_info(logger: logging.Logger):
    """
    Log system information
    
    Args:
        logger: Logger instance
    """
    import platform
    import torch
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    log_gpu_info(logger)
    
    logger.info("=" * 60)


def log_config(logger: logging.Logger, config: dict):
    """
    Log configuration dictionary
    
    Args:
        logger: Logger instance
        config: Configuration dict
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    
    def log_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 60)


if __name__ == "__main__":
    """Test logging utilities"""
    
    # Setup logger
    logger = setup_logger(
        name='test',
        log_file='test.log',
        level=logging.INFO
    )
    
    logger.info("Testing logger...")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test system info
    log_system_info(logger)
    
    # Test config logging
    config = {
        'model': {
            'name': 'deit_tiny',
            'num_classes': 51
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 5e-5
        }
    }
    
    log_config(logger, config)
    
    # Test experiment dir creation
    exp_dir = create_experiment_dir()
    logger.info(f"Created experiment directory: {exp_dir}")
