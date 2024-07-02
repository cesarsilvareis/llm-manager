import yaml, shutil
from src import get_actual_path
from src.model import ModelConfig, CurrentModel, DownloadStatus
from huggingface_hub import snapshot_download
from pathlib import Path
from src.logger import get_logger
from pandas import read_csv
from src.inference import Prompt, PromptType, RawPrompt, CompletionPrompt, SystemPrompt, Template, ModelExecution, Inference


logger = get_logger(__name__)

def load_model_from_fs(filename: str) -> ModelConfig:
    source = get_actual_path(filename, mode="model")

    with source.open("r", encoding="utf-8") as s:
        model_config = yaml.safe_load(s)

    return ModelConfig(filename, **model_config)
    

def load_model_from_hf(model: ModelConfig):

    if not CurrentModel.on(): # TODO improve this...
        import os
        allmodels = os.listdir("./configs/models/")
        for m in allmodels:
            cfg = load_model_from_fs(Path(m).stem)
            if cfg.status == DownloadStatus.COMPLETED  and cfg.local == CurrentModel.LOCAL:
                CurrentModel.initiate(cfg)
                break

    destination: Path = get_actual_path(fileOrDir=model.local, mode="store")

    # This is enough to just download one time for correcting consequents
    if model.status != DownloadStatus.UNINITIALIZED and model.local == CurrentModel.LOCAL \
        and CurrentModel.on() and not CurrentModel.included(model):
            model.invalidate_local()
    
    match model.status:
        case DownloadStatus.COMPLETED:
            logger.debug(f"No downloaded model '{model}' as it is present on '{destination}'")
            return
        case DownloadStatus.STARTED:
            assert destination.exists()
            logger.debug(f"Continuing downloading model '{model}' in destination '{destination}'...")

        case DownloadStatus.UNINITIALIZED:
            if destination.exists(): # Disk space preparation
                shutil.rmtree(destination)
            destination.mkdir()

            model.start_download()
            
            logger.debug(f"Cleaned directory '{destination}' for downloading model '{model}'...")
        case _: # NEVER
            logger.warn(f"Download status not recognized")


    if model.local == CurrentModel.LOCAL:
        if not CurrentModel.on():
            CurrentModel.initiate(model)
        else:
            if not CurrentModel.included(model):
                CurrentModel.invalidate(new_modelcfg=model)
    
    logger.info(f"Downloading model '{model['name']}' from '{model['hf_repo']}'")
    downloaded_path = snapshot_download(
        repo_id=model['hf_repo'], 
        repo_type="model",
        revision=model['revision'],
        local_dir=destination,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
        allow_patterns=[model["weightpat"], "*config*", "*tokenizer*", "*json"]
    )
    logger.info(f"Model '{model['name']}' was stored in '{downloaded_path}'")
    model.validate_local()



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


# Load executions sorted by models for reusability
def load_executions(path: str|Path, output_filename: str|None=None) -> list[ModelExecution]:
    executions: list[ModelExecution] = list()

    if isinstance(path, str):
        path = Path(path)
    
    assert path.exists()
    df = read_csv(path)
    df.set_index("execution", inplace=True)
    groups = df.groupby(by="model", sort=False)

    basefilename = output_filename if output_filename is not None else path.stem

    for model, model_df in groups:
        loaded_model = load_model_from_fs(model)
        for index, row in model_df.iterrows():
            if row["type"].lower() == "test":   # TODO
                continue

            executions.append(Inference(index, 
                model=loaded_model,
                prompt=load_prompt(row["input"]), 
                output_filename=f"{basefilename}_{index}"
            ))
    
    return executions

def update_config(modelcfg: ModelConfig):
    cfg_file = get_actual_path(fileOrDir=modelcfg.filename, mode="model")
    with cfg_file.open("w") as c:
        yaml.dump(dict(modelcfg), c)
