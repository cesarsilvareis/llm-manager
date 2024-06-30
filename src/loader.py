import yaml
from src import get_actual_path
from src.inference import *
from src.model import ModelConfig



def load_model_from_fs(filename: str) -> ModelConfig:
    ...


def load_model_from_hf() -> ModelConfig:
    ...
    return load_model_from_fs()


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