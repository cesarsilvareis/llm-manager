from loader import ModelConfig
from prompt import Prompt
from abc import ABC, abstractmethod
from typing import Self, Optional
from src import get_actual_path
from transformers import AutoModel

class ModelExecution(ABC):

    ACTIVIVE_EXECUTIONS: set[int]=set()

    def __init__(self, id: int, model: ModelConfig, output_filename: str):
        assert id not in self.ACTIVIVE_EXECUTIONS
        ModelExecution.ACTIVIVE_EXECUTIONS.add(id)
        
        self._id = id
        self._model = model
        self._result = None
        self._saved = False
        self._output_filename = output_filename

    @property
    def id(this) -> int:
        return this._id
    
    @property
    def model(this) -> str:
        return this._model["name"]

    @property
    def result(this) -> Optional[str]:
        return this._result

    @property
    def saved(this) -> bool:
        return this._saved
    
    @property
    def output_filename(this) -> str:
        return this._output_filename

    def run(self) -> str:
        self.setup()
        self.execute()
        self.teardown()

    @abstractmethod
    def setup(self):
        raise NotImplementedError
    
    @abstractmethod
    def execute(self):
        raise NotImplementedError
    
    @abstractmethod
    def teardown(self):
        raise NotImplementedError
    
    def save(self: Self):
        assert self.result is not None

        outputfile = get_actual_path(self.output_filename, mode="result")

        with outputfile.open("w") as o:
            o.write(self.result.encode("utf-8"))


    def __repr__(self, **extra_params) -> str:
        parameters = {
            "model": self.model,
            **extra_params,
            "saved": self.saved,
            "output_filename": self.output_filename
        }
        return (
            f"{self.__class__.__name__}(" +
            "; ".join(f"{k}='{v}'" for k, v in parameters.items())
            + ")"
        )


class Inference(ModelExecution):

    def __init__(self, id: int, model: ModelConfig, prompt: Prompt):
        super().__init__(id, model)
        self._prompt = prompt
        self._result: str = ""

    @property
    def prompt(this) -> Prompt:
        return this._prompt
    
    def setup(self: Self):
        return super().setup()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prompt={self.prompt}; "
            f"model={self.model})"
        )