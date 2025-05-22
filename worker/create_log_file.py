from pathlib import Path
from config.load_config import LOG_PATH, PROGRAM_LOG
from log_manager import log


def touch_files():
    """
    Touches (creates or updates modification time) for files specified
    by LOG_PATH and PROGRAM_LOG environment variables.
    """
    log_path_str = LOG_PATH
    program_log_str = PROGRAM_LOG

    if log_path_str:
        Path(log_path_str).touch()
        log.info(f"Touched: {log_path_str}")
    else:
        log.warning("Warning: LOG_PATH environment variable not set.")

    if program_log_str:
        Path(program_log_str).touch()
        log.info(f"Touched: {program_log_str}")
    else:
        log.warning("Warning: PROGRAM_LOG environment variable not set.")
