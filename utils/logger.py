# ─────────────────────────────────────────────
# utils/logger.py — Centralized logging setup
# ─────────────────────────────────────────────

import logging
import os
from config import LOG_LEVEL, LOG_DIR

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    Logs to both console and a file in data/logs/.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(LOG_DIR, "evacuation_system.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
