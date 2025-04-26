# import logging
# import os,sys
# from datetime import datetime

# LOG_DIR = "logs"
# LOG_DIR = os.path.join(os.getcwd(),LOG_DIR)

# os.makedirs(LOG_DIR,exist_ok=True)



# CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
# file_name = f"log_{CURRENT_TIME_STAMP}.log"

# # output should be logs/log_2025-02-19_10-30-00.log

# log_file_path = os.path.join(LOG_DIR,file_name)

# logging.basicConfig(filename=log_file_path,
#                     filemode="w",
#                     format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
#                     level=logging.INFO)


# import logging
# import os
# from datetime import datetime

# # Define logs directory
# LOG_DIR = "logs"
# LOG_PATH = os.path.join(os.getcwd(), LOG_DIR)
# os.makedirs(LOG_PATH, exist_ok=True)

# # Create a timestamped log file
# CURRENT_TIME_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
# LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)

# # Configure logging
# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     filemode="w",
#     format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )

# # Create a logger instance
# logger = logging.getLogger(__name__)


# --- logger_config.py ---

# --- src/AIAnswerEvaluationSystem/logger.py ---
# (Or src/AIAnswerEvaluationSystem/logger/__init__.py)

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys
from typing import Optional # Import Optional for type hinting

# Define default constants *outside* the function for clarity
DEFAULT_LOG_DIR = "logs"
DEFAULT_ENV_VAR_NAME = "LOG_LEVEL"
DEFAULT_LEVEL_STR = "INFO"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024 # 10 MB
DEFAULT_BACKUP_COUNT = 5

# --- setup_logging function ---
def setup_logging(
    log_dir: str = DEFAULT_LOG_DIR,
    log_level_env_var: str = DEFAULT_ENV_VAR_NAME,
    default_log_level: str = DEFAULT_LEVEL_STR,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT
    ) -> logging.Logger:
    """
    Configures logging to output to both a rotating file and the console.
    Should be called ONCE at application startup.
    """
    try:
        # --- Determine Log Level ---
        # Explicitly reference the arguments received by the function
        env_var_name = log_level_env_var
        default_level = default_log_level
        print(f"[DEBUG log setup] Env Var Name: '{env_var_name}', Default Level: '{default_level}'") # Debug print

        # Check if the inputs are strings
        if not isinstance(env_var_name, str) or not isinstance(default_level, str):
            raise TypeError("Log level variables must be strings")

        log_level_str_from_env = os.environ.get(env_var_name)
        print(f"[DEBUG log setup] Value from os.environ.get('{env_var_name}'): {log_level_str_from_env}") # Debug print

        if log_level_str_from_env:
            log_level_str = log_level_str_from_env.upper()
            print(f"[DEBUG log setup] Using Level from ENV: '{log_level_str}'") # Debug print
        else:
            log_level_str = default_level.upper()
            print(f"[DEBUG log setup] Using Default Level: '{log_level_str}'") # Debug print

        log_level = getattr(logging, log_level_str, logging.INFO) # Fallback safely to INFO if conversion fails
        print(f"[DEBUG log setup] Resolved logging Level: {log_level} ({logging.getLevelName(log_level)})") # Debug print


        # --- Create Log Directory ---
        log_path = os.path.join(os.getcwd(), log_dir)
        # print(f"[DEBUG log setup] Log Directory Path: {log_path}") # Debug print (optional)
        os.makedirs(log_path, exist_ok=True) # Use default mode


        # --- Create Timestamped Filename ---
        # Using daily rotation is generally preferred over per-second/minute
        current_time_stamp = datetime.now().strftime('%Y-%m-%d')
        log_file_name = f"app_log_{current_time_stamp}.log" # Base name for rotation
        log_file_path = os.path.join(log_path, log_file_name)
        # print(f"[DEBUG log setup] Log File Path: {log_file_path}") # Debug print


        # --- Configure Root Logger ---
        root_logger = logging.getLogger() # Get the root logger
        # Check if handlers were ALREADY added in this run (prevents duplicates)
        # Basic check - relies on specific handler names if refining needed
        if any(isinstance(h, (RotatingFileHandler, logging.StreamHandler)) for h in root_logger.handlers):
            print("[DEBUG log setup] Handlers seem to be already configured. Skipping reconfiguration.") # Debug
            # Alternatively, clear them if re-configuration is desired on subsequent calls (not recommended):
            # root_logger.handlers.clear()
        else:
             print("[DEBUG log setup] Configuring root logger handlers...") # Debug print
             root_logger.setLevel(log_level) # Set the minimum level the logger will handle

             log_formatter = logging.Formatter(
                 "[%(asctime)s] %(name)s:%(lineno)d - %(levelname)s - %(message)s",
                 datefmt="%Y-%m-%d %H:%M:%S"
             )

             # --- Rotating File Handler ---
             try:
                file_handler = RotatingFileHandler(
                    filename=log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
                )
                file_handler.setFormatter(log_formatter)
                # Individual handlers can also have levels set
                # file_handler.setLevel(logging.DEBUG) # Example: File logs everything
                root_logger.addHandler(file_handler)
             except Exception as fh_e:
                print(f"Error setting up file handler '{log_file_path}': {fh_e}", file=sys.stderr)

             # --- Console Handler ---
             console_handler = logging.StreamHandler(sys.stdout)
             console_handler.setFormatter(log_formatter)
             # console_handler.setLevel(logging.INFO) # Example: Console logs only INFO and above
             root_logger.addHandler(console_handler)

             root_logger.info(f"Logging configured. Level: {log_level_str}. File: {log_file_path}")

        return root_logger

    except Exception as setup_e:
        # Catch any error during setup itself
        print(f"CRITICAL ERROR during logging setup: {setup_e}", file=sys.stderr)
        # Fallback to basic console logging if setup fails catastrophically
        logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(levelname)s - %(message)s")
        logging.error("Falling back to basic console logging due to setup error.")
        return logging.getLogger() # Return the basic configured logger


# --- Call setup_logging ONCE when this module is imported ---
# Ensure defaults are used or override if needed.
# Handles the immediate configuration.
setup_logging(log_dir="logs", log_level_env_var="LOG_LEVEL", default_log_level="INFO")


# --- Create and EXPORT the application-specific logger instance ---
# Get a logger named after your main application package/module
logger = logging.getLogger("AIAnswerEvaluationSystem")
# It automatically inherits handlers and levels configured on the root logger by setup_logging()
logger.info("Logger 'AIAnswerEvaluationSystem' obtained and ready for import.")