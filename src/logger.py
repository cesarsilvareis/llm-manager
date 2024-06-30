# src/logger.py

import os, yaml
from logging import Logger, getLogger, basicConfig 
from logging.config import dictConfig
from pathlib import Path
from dotenv import load_dotenv
from src import ROOT_DIR

load_dotenv(ROOT_DIR.joinpath(".env"))

def setup_logging(configfile: str|Path):

    if isinstance(configfile, str):
        configfile = Path(configfile)

    """Setup logging configuration"""
    if configfile.is_file():
        with configfile.open("r") as f:
            config = yaml.safe_load(f.read())
        dictConfig(config)
    else:
        default_level = os.getenv("LOG_LEVEL")
        basicConfig(level=default_level)
        print(f"Warning: Logging configuration file not found at {configfile}. Using default level: {default_level}")

def get_logger(callerfile: str) -> Logger:
    return getLogger(callerfile)


