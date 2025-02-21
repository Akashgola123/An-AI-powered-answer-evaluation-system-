
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "AIAnswerEvaluationSystem"

list_of_files = [
    ".github/workflows/.gitkeep",
    "research/trials.ipynb",
    f"src/{project_name}/agents/agentss.py",
    f"src/{project_name}/prompt/prompts.py",
    f"src/{project_name}/llms/llmss.py",
    f"src/{project_name}/Graph_Database/Graph_Databases.py",
    f"src/{project_name}/Memory_Manager/Memory_Managers.py",
    f"src/{project_name}/utils/utils.py",
    f"src/{project_name}/telemetry/telemetrys.py",
    f"src/{project_name}/config/configs.py",
    f"src/{project_name}/dataloader/dataloaders.py",
    f"src/{project_name}/retrievers/retrieverss.py",
    f"src/{project_name}/Score_Manager/Score_Managers.py",
    f"src/{project_name}/nlogger/__init__.py",
    f"src/{project_name}/main.py",
    "requirements.txt",
    "setup.py",
    "exception.py",
    "logs.py",
    "tests/test_main.py",
    "Makefile",
    "templates/index.html"


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