
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "AIAnswerEvaluationSystem"

list_of_files = [
    ".github/workflows/.gitkeep",
    "research/trials.ipynb",
    f"src/{project_name}/prompts.py",
    f"src/{project_name}/llmss.py",
    f"src/{project_name}/neo4j_Manager.py",
    f"src/{project_name}/ocr.py",
    f"src/{project_name}/Fetching.py",
    f"src/{project_name}/database_login_rigister.py",
    f"src/{project_name}/main.py",
    "requirements.txt",
    "setup.py",
    "exception.py",
    "logs.py",
    "tests/test_main.py",
    "Makefile",
    "fontend/index.html",
    ".env"


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