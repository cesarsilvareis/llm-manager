#!.venv/bin/python3.11

GPU_DEVICES = [0]

import os, yaml, dotenv
from argparse import ArgumentParser
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in GPU_DEVICES)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextGenerationPipeline, pipeline
from torch import float16

dotenv.load_dotenv(".env")

def get_model_from_config(model_config) -> TextGenerationPipeline:
    return pipeline(
        task="text-generation",
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config['save_dir'],
            device_map="auto",
            # offload_folder="offload",
            # torch_dtype=torch.float16
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_config['save_dir'],
            padding_side="left"
        ),
        num_workers=4,
        torch_dtype="auto",
        do_sample=True,
        **model_config['kwargs']
    )


def small_experiment(model):   # one-time run

    llm = get_model_from_config(model)
    print(f"""Ready to run LLM:
    -name: {model['name']}
    -gpu: {llm.device}
    -task: {llm.task}
    -dtype: {llm.torch_dtype}
    -size: {round(llm.model.num_parameters() / 1e9, 2)} B
    -max_input_sequence: {round(llm.tokenizer.model_max_length / 1e9, 2)} B
    -special_tokens: {llm.tokenizer.all_special_tokens}
    """)
    prompt = input(f"Enter your one-time prompt to {model['name']}\n> ")

    response = llm(prompt)[0]['generated_text']
    print(f"\n{model['name']}:\n{response}")


def main(current_model):
    small_experiment(model=current_model)


if __name__ == "__main__":

    parser = ArgumentParser(description="Loads a model and sets it as current active")
    parser.add_argument("-m", "--model", type=str, required=True, 
        help="It's your model configuration (a personalized YAML file in your FS)"
    )

    args = parser.parse_args()
    
    with open(args.model, 'r') as m:
        model = yaml.safe_load(m)
    main(model)
