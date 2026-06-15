import sys
from loguru import logger
from .config import config

logger.remove()
fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
logger.add(sys.stderr, format=fmt, level=config.LOG_LEVEL)
