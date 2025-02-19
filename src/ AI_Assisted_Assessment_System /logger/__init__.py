import logging 
import os,sys
from datetime import datetime

LOG_DIR = "logs"

LOG_DIR = os.path.join(os.getcwd(), LOG_DIR)

os.makedirs(LOG_DIR, exist_ok=True)


# .log and current time stamp for the log file

CUURENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
FILE_NAME =  f"{CUURENT_TIME_STAMP}.log"


### output should be log_2023-08-22_12-55-30.log

LOG_FILE = os.path.join(LOG_DIR, FILE_NAME)


logging.basicConfig(filename=LOG_FILE, filemode="w", format="[%(asctime)s]: %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.info("Logging is set up")
logging.info(f"Log file is saved at : {LOG_FILE}")