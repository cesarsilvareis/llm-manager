from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR.joinpath(".env"))

def get_actual_path(fileOrDir: str, mode: Literal["model", "prompt", "test", "logs", "result", "store"]) -> Path:
    match mode:
        case "model":
            path, ext = ROOT_DIR.joinpath("configs", "models"), "yml"
        case "prompt":
            path, ext = ROOT_DIR.joinpath("configs", "prompts"), "yml"
        # case "test":
            ...
        case "store":
            path, ext = ROOT_DIR.joinpath(".storage"), None
        case "log":
            path, ext = ROOT_DIR.joinpath("outputs", "logs"), "yml"
        case "result":
            path, ext = ROOT_DIR.joinpath("outputs", "results"), "txt"
        case _:
            raise ValueError(f"[ERROR] Unknown mode '{mode}' for getting the actual \
                             path for the file '{fileOrDir}'.")
    
    assert path.exists(), f"[ERROR] Built path '{path}' is not actual as it does not exist." 
        
    return path.joinpath(Path(f"{fileOrDir}.{ext}")) if ext else path.joinpath(fileOrDir) 
