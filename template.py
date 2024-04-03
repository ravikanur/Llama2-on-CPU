import os
from pathlib import Path
from datetime import datetime
import logging

def create_log_file():
        LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d')}.log"
        LOG_FILE_DIR_PATH = './logs'
        os.makedirs(LOG_FILE_DIR_PATH, exist_ok=True)
        LOG_FILES = os.listdir(LOG_FILE_DIR_PATH)
        #if LOG_FILE not in LOG_FILES:

            #os.makedirs(os.path.join(LOG_FILE_DIR_PATH, LOG_FILE).replace("\\", "/"))  
        LOG_FILE_PATH = os.path.join(LOG_FILE_DIR_PATH, LOG_FILE).replace("\\", "/")
        return LOG_FILE_PATH

logging.basicConfig(filename=create_log_file(), level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    "src/__init__.py",
    "src/run_local.py",
    "src/helper.py",
    "model/instruction.txt",
    "requirements.txt",
    "setup.py",
    "main.py",
    "research/trials.ipynb",

]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")