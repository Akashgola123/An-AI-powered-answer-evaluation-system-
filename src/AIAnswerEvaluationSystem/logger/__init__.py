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


import logging
import os
from datetime import datetime

# Define logs directory
LOG_DIR = "logs"
LOG_PATH = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(LOG_PATH, exist_ok=True)

# Create a timestamped log file
CURRENT_TIME_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Create a logger instance
logger = logging.getLogger(__name__)
