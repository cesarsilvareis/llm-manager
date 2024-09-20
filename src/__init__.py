from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR.joinpath(".env"))

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Limit GPU memory allocation to a fraction of total memory
# Adjust `0.5` to the fraction of GPU memory you want to allocate
# for device in physical_devices:
#     tf.config.experimental.set_virtual_device_configuration(
#         device,
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(0.2 * 1024))]
#     )


def get_actual_path(fileOrDir: str, mode: Literal["config", "prompt", "data", "log", "result", "model"]) -> Path:
    match mode:
        case "config":
            path, ext = ROOT_DIR.joinpath("configs", "models"), "yml"
        case "prompt":
            path, ext = ROOT_DIR.joinpath("configs", "prompts"), "yml"
        case "model":
            path, ext = ROOT_DIR.joinpath(".storage", "models"), None
        case "data":
            path, ext = ROOT_DIR.joinpath(".storage", "data"), None
        case "log":
            path, ext = ROOT_DIR.joinpath("outputs", "logs"), None
        case "result":
            path, ext = ROOT_DIR.joinpath("outputs", "results"), "txt"
        case _:
            raise ValueError(f"[ERROR] Unknown mode '{mode}' for getting the actual \
                             path for the file '{fileOrDir}'.")
    
    assert path.exists(), f"[ERROR] Built path '{path}' is not actual as it does not exist." 
        
    return path.joinpath(Path(f"{fileOrDir}.{ext}")) if ext else path.joinpath(fileOrDir)
