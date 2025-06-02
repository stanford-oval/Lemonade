import warnings

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

warnings.filterwarnings(
    "ignore", message="Valid config keys have changed in V2:.*", category=UserWarning
)

# Remove the default loguru handler
logger.remove()

# Add a new handler using RichHandler for console output
logger.add(
    RichHandler(
        console=Console(markup=True, soft_wrap=False),
        show_time=False,
        rich_tracebacks=True,
    ),  # Enable rich markup for colored output
    level="INFO",  # Set the logging level
    format="{message}",
    backtrace=True,  # Include the backtrace in the log
    diagnose=True,  # Include diagnostic information in the log
)

# Add another handler for saving debug logs to a file
logger.add(
    "debug_logs.log",  # File path for the log file
    level="DEBUG",  # Set the logging level to DEBUG
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",  # Log format
    rotation="10 MB",  # Rotate the log file when it reaches 10 MB
    retention="1 hour",  # Retain log files for 7 days
    backtrace=True,  # Include the backtrace in the log
    diagnose=True,  # Include diagnostic information in the log
)
