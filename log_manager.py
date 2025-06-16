import os
import logging
import time
import sys
import config.load_config as CONFIG

from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install


log = None

logfile = CONFIG.LOG_PATH


def setup_logging(clean=False, debug=False):
    global log

    if log is not None:
        return log

    try:
        if clean and os.path.isfile(logfile):
            os.remove(logfile)
        time.sleep(0.1)  # prevent race condition
    except:
        pass

    # Create logger
    log = logging.getLogger("sd")
    log.setLevel(logging.DEBUG)

    # Clear any existing handlers
    log.handlers.clear()

    # File handler
    # file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")

    # file_handler.setLevel(logging.DEBUG)
    # file_formatter = logging.Formatter(
    #     "%(asctime)s | %(levelname)s | %(pathname)s | %(message)s"
    # )
    # file_handler.setFormatter(file_formatter)
    # log.addHandler(file_handler)

    # Console setup for Rich
    console = Console(
        log_time=True,
        log_time_format="%H:%M:%S-%f",
        theme=Theme(
            {
                "traceback.border": "black",
                "traceback.border.syntax_error": "black",
                "inspect.value.border": "black",
            }
        ),
    )
    pretty_install(console=console)
    traceback_install(
        console=console,
        extra_lines=1,
        width=console.width,
        word_wrap=False,
        indent_guides=False,
        suppress=[],
    )

    # Rich console handler for stdout
    rh = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format="%H:%M:%S-%f",
        level=logging.DEBUG if debug else logging.INFO,
        console=console,
    )
    rh.set_name(logging.DEBUG if debug else logging.INFO)
    log.addHandler(rh)

    return log


log = setup_logging()
