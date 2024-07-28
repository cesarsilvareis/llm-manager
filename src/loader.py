import yaml, shutil
from src import get_actual_path
from src.model import ModelConfig, CurrentModel, DownloadStatus
from huggingface_hub import snapshot_download
from pathlib import Path
from src.logger import get_logger
from pandas import read_csv
from src.execution import ModelExecution, ModelBatch
from src.inference import Prompt, PromptType, RawPrompt, CompletionPrompt, SystemPrompt, Template, Inference, Testing
from src.evaluation import TruthfulQA, PubMedSummary, ClinicalParaph
from datasets import Dataset, DatasetDict, load_from_disk
from src.tasks import ED_MCQ

logger = get_logger(__name__)

def load_modelcfg_from_fs(filename: str) -> ModelConfig:
    source = get_actual_path(filename, mode="config")

    with source.open("r", encoding="utf-8") as s:
        model_config = yaml.safe_load(s)

    return ModelConfig(filename, **model_config)
    
def update_config(modelcfg: ModelConfig):
    cfg_file = get_actual_path(fileOrDir=modelcfg.filename, mode="config")
    with cfg_file.open("w") as c:
        yaml.dump(dict(modelcfg), c)


def load_model_from_hf(model: ModelConfig):

    if not CurrentModel.on(): # TODO improve this...
        import os
        allmodels = os.listdir("./configs/models/")
        for m in allmodels:
            cfg = load_modelcfg_from_fs(Path(m).stem)
            if cfg.status == DownloadStatus.COMPLETED  and cfg.local == CurrentModel.LOCAL:
                CurrentModel.initiate(cfg)
                break

    # This is enough to just download one time for correcting consequents
    if model.status != DownloadStatus.UNINITIALIZED and model.local == CurrentModel.LOCAL \
        and CurrentModel.on() and not CurrentModel.included(model):
            model.invalidate_local()

    destination: Path = get_actual_path(fileOrDir=model.local, mode="model")

    match model.status:
        case DownloadStatus.COMPLETED:
            logger.debug(f"No downloaded model '{model['name']}' as it is present on '{destination}'")
            return
        case DownloadStatus.STARTED:
            assert destination.exists()
            logger.debug(f"Continuing downloading model '{model['name']}' in destination '{destination}'...")

        case _: # DownloadStatus.UNINITIALIZED or others
            if destination.exists(): # Disk space preparation
                shutil.rmtree(destination)
            destination.mkdir()

            model.start_download()
            
            logger.debug(f"Cleaned directory '{destination}' for downloading model '{model['name']}'...")


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
        resume_download=True,
        max_workers=4,
        allow_patterns=[model["weightpat"], "*config*", "*tokenizer*", "*json"]
    )
    logger.info(f"Model '{model['name']}' was stored in '{downloaded_path}'")
    model.validate_local()



def load_prompt_fs(filename: str) -> Prompt:
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


def load_data_from_fs(filename: str) -> Dataset| DatasetDict:
    data_local = get_actual_path(filename, mode="data")
    return load_from_disk(data_local)

# Load executions sorted by models for reusability
def load_executions(path: str|Path, output_filename: str|None=None, batched: bool=True) -> list[ModelExecution]|list[ModelBatch]:
    executions: list[ModelExecution] = list()

    if isinstance(path, str):
        path = Path(path)
    
    assert path.exists()
    df = read_csv(path)
    df.set_index("execution", inplace=True)
    groups = df.groupby(by="model", sort=False)

    basefilename = output_filename if output_filename is not None else path.stem

    loaded_prompts = dict()
    loaded_tests = dict()

    for model, model_df in groups:
        loaded_model = load_modelcfg_from_fs(model)
        for index, row in model_df.iterrows():
            match row["type"].lower():
                case "inference":
                    if row["input"] not in loaded_prompts:
                        loaded_prompts[row["input"]] = load_prompt_fs(row["input"])
                    execution = Inference(index, modelcfg=loaded_model,
                        prompt=loaded_prompts[row["input"]], 
                        output_filename=f"infer_{basefilename}_{index}")
                case "benchmark":
                    match row["input"].lower():
                        case "truthfulqa":
                            execution = TruthfulQA(index, modelcfg=loaded_model,
                                outputfile=f"truthful_{basefilename}_{index}", save_latents="metric")
                        case "pubmedsum":
                            execution = PubMedSummary(index, modelcfg=loaded_model,
                                outputfile=f"pubmedsum_{basefilename}_{index}", save_latents="metric")
                        case "clinicalparaph":
                            execution = ClinicalParaph(index, modelcfg=loaded_model,
                                outputfile=f"clinpar_{basefilename}_{index}", save_latents="metric")
                        case _:
                            raise ValueError(f"Unknown benchmark name '{row['input']}' as the input column value.")
                case "testing":
                    if row["input"] not in loaded_tests:
                        loaded_tests[row["input"]] = load_data_from_fs(row["input"])
                    execution = Testing(index, modelcfg=loaded_model,
                        data_local=loaded_tests[row["input"]], task=ED_MCQ(),
                        output_filename=f"test_{basefilename}_{index}")
                case _:
                    raise ValueError(f"Unknown execution type '{row['type']}'.")

            executions.append(execution)
    
    return ModelBatch.from_list(executions) if batched else executions


