from pathlib import Path
from typing import Literal

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_actual_path(filename: str, mode: Literal["model", "prompt", "test", "logs", "result"]) -> Path:
    match mode:
        case "model":
            path, ext = ROOT_DIR.joinpath("configs", "models"), "yml"
        case "prompt":
            path, ext = ROOT_DIR.joinpath("configs", "prompts"), "yml"
        # case "test":
            ...
        case "log":
            path, ext = ROOT_DIR.joinpath("outputs", "logs"), "yml"
        case "result":
            path, ext = ROOT_DIR.joinpath("outputs", "results"), "txt"
        case _:
            raise ValueError(f"[ERROR] Unknown mode '{mode}' for getting the actual \
                             path for the file '{filename}'.")
    
    assert path.exists(), "[ERROR] Built path is not actual as it does not exist." 
        
    return path.joinpath(Path(f"{filename}.{ext}"))

from src.inference import *
# TODO: training...
