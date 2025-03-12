"""
utils/logger.py

This module provides a custom logger for standardized debugging output.
It defines functions like log_debug(), log_info(), log_warning(), and log_error() which print messages
to both the console and a log file ("project.log") with a consistent format.
"""

import logging
import sys

# Create a logger instance
logger = logging.getLogger("MultiAgentLogger")
logger.setLevel(logging.DEBUG)

# Create handlers: one for console output and one for file output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("project.log")
file_handler.setLevel(logging.DEBUG)

# Define a formatter and set it for both handlers
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def log_debug(message):
    logger.debug(message)

def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)
