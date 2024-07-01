#!.venv/bin/python3.11

import os, yaml, shutil
from argparse import ArgumentParser, BooleanOptionalAction
from huggingface_hub import snapshot_download

from experiment_model import small_experiment

MODELS_DIR = "./models"

def get_models_configs() -> list[str]:
    return list(map(lambda c: os.path.join(MODELS_DIR, c), 
        filter(lambda f: f.endswith(".yml"), 
                os.listdir(MODELS_DIR))
    ))


def invalidate_model_config(model, config):
    print(f"Unvalidated {model['name']}")

    model["saved"] = False
    model["save_dir"] = ""
    with open(config, "w") as o:
        yaml.dump(model, o)    


def update_configs(model, exclude_dir):
    models_config = get_models_configs()
 
    for config in models_config:
        with open(config, "r") as o:
            other = yaml.safe_load(o)

        if other == model or not other["saved"] or \
                (os.path.exists(other["save_dir"]) and not os.path.samefile(other["save_dir"], exclude_dir)):
            continue

        invalidate_model_config(other, config)


def overwrite_directory(model, dir) -> tuple[bool, str | None]:
    print(f"Warning: Overwriting directory {dir}...")
    update_configs(model, dir)

    # Reset storage directory 
    if not os.path.exists(dir):
        return False, f"Storage directory '{dir}' does not exist to be overwritten."  # should never go here
    shutil.rmtree(dir)

    return True,


def makes_directory_compatible(model, dir, overwrite=False) -> tuple[bool, str | None]:
    # Check model already exist in another destiny
    if model['saved']:
        if not os.path.samefile(model['save_dir'], dir):
            return False, f"Model can not be downloaded in multiple destinations. Indeed the model encounters in {model['save_dir']}."
        return True,
    
    if overwrite:
        return overwrite_directory(model, dir)

    # Check overwrites among other models for the same directory
    models_config = get_models_configs()
    for config in models_config:
        with open(config, "r") as o:
            other = yaml.safe_load(o)

        if other['saved']:
            if not os.path.exists(other['save_dir']):
                invalidate_model_config(other, config)
                continue

            if os.path.samefile(other["save_dir"], dir):
                return False, f"Model {other['name']} is already saved in the given directory."

    return True,


def main(model_config: dict, output_dir, overwrite: bool):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        update_configs(model_config, output_dir)
        overwrite = False

    with open(model_config, "r") as m:
        model = yaml.safe_load(m)

    status = makes_directory_compatible(model, output_dir, overwrite)
    if not status[0]:
        raise ValueError(f"[ERROR] {status[1]}")

    print(f"Downloading model {model['name']} from {model['hf_repo']}...")

    # Before starting keep saved state for resume download feature!
    model["saved"] = True
    model["save_dir"] = output_dir
    with open(model_config, "w") as m:
        yaml.dump(model, m)

    # https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/en/guides/download

    downloaded_path = snapshot_download(
        repo_id=model['hf_repo'], 
        repo_type="model",
        revision=model['revision'],
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
        ignore_patterns=["*.png", "*.jpg", "*safetensors*"]
    )

    print(f"\nModel weights stored in {downloaded_path}")

    if input("Would you like to make a small experiment with the model? (y)/n:").lower() != 'n':
        small_experiment(model)


if __name__ == "__main__":

    parser = ArgumentParser(description="Loads a model and sets it as current active")
    parser.add_argument("-m", "--model", type=str, required=True, 
                        help="It's your input model (a YAML config file)")
    parser.add_argument("-d", "--dir", type=str, required=True, 
                        help="Intended storage directory")
    parser.add_argument("--overwrite", action=BooleanOptionalAction, default=False, 
                        help="Would you like to overwrite existing models?")

    args = parser.parse_args()

    main(args.model, args.dir, args.overwrite)
