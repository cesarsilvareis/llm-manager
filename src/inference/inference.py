from src import get_actual_path
from src.execution import ModelExecution
from src.model import ModelConfig
from src.inference import Prompt
from itertools import product
from typing import Self, Any
from transformers import pipeline

class Inference(ModelExecution):

    def __init__(self, id: int, modelcfg: ModelConfig, output_filename: str, prompt: Prompt):
        super().__init__(id, modelcfg, output_filename)
        self._prompt = prompt

    @property
    def prompt(this) -> Prompt:
        return this._prompt
    
    def setup(self: Self):
        return {
            "caller": pipeline,
            "task": "text-generation",
            "model": get_actual_path(self.modelcfg.local, mode="model"),
            "num_workers": 4,
            "torch_dtype":"auto",
            "trust_remote_code": False,
            "device_map": "auto",
            "model_kwargs": self.modelcfg["model_params"],
        }

    def execute(self, model, gen_params: dict[str, Any]) -> list[str]:
        return model(
            str(self.prompt), 
            do_sample=True, 
            return_full_text=False, 
            **gen_params)[0]['generated_text']

    def __repr__(self, **_) -> str:
        return super().__repr__(prompt=self.prompt.name)