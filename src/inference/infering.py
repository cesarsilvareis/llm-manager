from src import get_actual_path
from src.execution import ModelExecution
from src.model import ModelConfig
from src.inference import PromptType, Prompt
from typing import Self, Any
from transformers import pipeline, TextGenerationPipeline

class Inference(ModelExecution):

    def __init__(self, id: int, modelcfg: ModelConfig, output_filename: str, prompt: Prompt):
        super().__init__(id, modelcfg, f"inference/{output_filename}")
        self._prompt = prompt
    
    def setup(self: Self):
        return {
            "caller": pipeline,
            "task": "text-generation",
            "model": get_actual_path(self.modelcfg.local, mode="model"),
            "num_workers": 4,
            "trust_remote_code": False,
            "model_kwargs": self.modelcfg["model_params"],
        }

    def execute(self, generator: TextGenerationPipeline, gen_params: dict[str, Any]) -> list[str]:
        prompt = generator.tokenizer.apply_chat_template([
            {"role": "system", "content": self._prompt.system_message},
            {"role": "user", "content": self._prompt.human_message}
        ], tokenize=False, add_generation_prompt=True) if self._prompt.type == PromptType.CONTROL else self._prompt 

        return generator(
            prompt,
            return_full_text=False, num_return_sequences=1,
            **gen_params)[0]['generated_text']

    def __repr__(self, **_) -> str:
        return super().__repr__(prompt=self.prompt.name)