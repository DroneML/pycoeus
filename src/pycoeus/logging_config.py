import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from logging import handlers

import numpy as np
import pandas as pd


def setup_logger(logger: logging.Logger, stdout_level=logging.INFO):
    """
    Setup the input logger, add handlers and config logging level.
    """

    path = Path('log')
    path.mkdir(exist_ok=True, parents=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create handlers
    debug_fh = logging.handlers.RotatingFileHandler(path / 'debug.log', maxBytes=10 * 1024 * 1024, backupCount=5)
    debug_fh.setLevel(logging.DEBUG)
    debug_fh.setFormatter(formatter)
    info_fh = logging.handlers.RotatingFileHandler(path / 'info.log')
    info_fh.setLevel(logging.INFO)
    info_fh.setFormatter(formatter)
    stdout_sh = logging.StreamHandler(sys.stdout)
    stdout_sh.setLevel(stdout_level)
    stdout_sh.setFormatter(formatter)

    # Configure logger and add handlers
    logger.addHandler(debug_fh)
    logger.addHandler(info_fh)
    logger.addHandler(stdout_sh)

    return logger


@contextmanager
def log_duration(task_name: str, logger: logging.Logger):
    """
    Log duration of a task.
    """
    logger.info(f"{task_name} started")
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time
    logger.info(f"{task_name} finished in {duration:.4f} seconds")


def log_array(data: np.ndarray, logger, array_name:str="array") -> None:
    logger.debug(f"{array_name} shape: {data.shape}")
