import yaml
from src import get_actual_path
from src.inference import *
from src.model import ModelConfig, CurrentModel
from huggingface_hub import snapshot_download
from pathlib import Path
from src.logger import get_logger


logger = get_logger(__name__)

def load_model_from_fs(filename: str) -> ModelConfig:
    source = get_actual_path(filename, mode="model")

    with source.open("r", encoding="utf-8") as s:
        model_config = yaml.safe_load(s)

    return ModelConfig(**model_config)


def load_model_from_hf(model: ModelConfig):
    if model["local"] == CurrentModel.LOCAL: # It's enough just one download for correcting input locations 
        if CurrentModel.on():
            if not CurrentModel.included(model):
                model["local"] = ""
        else:
            CurrentModel.initiate(model)

    if model["local"] != "": return
    
    destination: Path = get_actual_path(fileOrDir=CurrentModel.LOCAL, mode="store")

    if not destination.exists():
        destination.mkdir()
        CurrentModel.initiate(model)

    logger.info(f"Downloading model '{model['name']}' from '{model['hf_repo']}'")
    downloaded_path = snapshot_download(
        repo_id=model['hf_repo'], 
        repo_type="model",
        revision=model['revision'],
        local_dir=destination,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
        ignore_patterns=["*.png", "*.jpg", "*safetensors*"]
    )
    logger.info(f"Model '{model['name']}' was stored in '{downloaded_path}'")
    
    CurrentModel.INSTANCE.invalidate(new_modelcfg=model)


def load_prompt(filename: str) -> Prompt:
    source = get_actual_path(filename, mode="prompt")

    with source.open("r", encoding="utf-8") as s:
        prompt_data = yaml.safe_load(s)
    
    match prompt_data["type"]:
        case PromptType.RAW:
            return RawPrompt(prompt_data["content"], prompt_data["name"], prompt_data["task"])
        case PromptType.COMPLETION:
            return CompletionPrompt(prompt_data["content"], prompt_data["name"])
        case PromptType.CONTROL:
            return SystemPrompt(prompt_data["name"], prompt_data["task"], prompt_data["system"], prompt_data["human"])
        case PromptType.PARAMETRIC:
            return Template(prompt_data["content"], prompt_data["name"], prompt_data["task"], prompt_data["params"])
        
    raise ValueError(f"[ERROR] File '{filename}' do not contain a valid prompt")