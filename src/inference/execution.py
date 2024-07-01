from loader import ModelConfig
from prompt import Prompt
from abc import ABC, abstractmethod
from typing import Self, Optional
from src import get_actual_path
from src.logger import get_logger
from src.loader import load_model_from_hf
from transformers import Pipeline, pipeline

logger = get_logger(__name__)

class ModelExecution(ABC):

    ACTIVIVE_EXECUTIONS: set[int]=set()
    KEY: str

    def __init__(self, id: int, model: ModelConfig, output_filename: str):
        assert id not in self.ACTIVIVE_EXECUTIONS
        ModelExecution.ACTIVIVE_EXECUTIONS.add(id)
        
        self._id = id
        self._model = model
        self._last_result = None
        self._output_filename = output_filename

    @property
    def id(this) -> int:
        return this._id
    
    @property
    def model(this) -> str:
        return this._model["name"]
    
    @property
    def last_result(this) -> Optional[str]:
        return this._last_result
    
    @property
    def output_filename(this) -> str:
        return f"{this._output_filename}_{this.id}"

    def run(self, single: bool=False) -> str:
        if (model := self._model["instances"].get(self.KEY, None)) is None:
            load_model_from_hf(self._model)
            model = self.setup()
        
        logger.info(f"Executing model '{self._model}'...")
        self._last_result = self.execute(model)
        
        logger.info(f"Saving result in '{self.output_filename}'...")
        self.save()

        if single:
            logger.info(f"Teardown execution {self.id} as single run")
            self.teardown()
        else:
            logger.debug(f"Remaining a teardown in the user side for executions on model '{self._model}' (key='{self.KEY}')")
            self._model.save_instance(model, key=self.KEY)
            logger.info(f"Saved model instance '{self._model}' for future runs")


    @abstractmethod
    def setup(self) -> Pipeline:
        raise NotImplementedError
    
    @abstractmethod
    def execute(self, model: Pipeline) -> str:
        raise NotImplementedError
    
    def teardown(self: Self):
        del self._model["instances"][self.KEY]  # remove model from memory
    
    def save(self: Self):
        assert self.last_result is not None

        outputfile = get_actual_path(self.output_filename, mode="result")

        with outputfile.open("w") as o:
            o.write(self.last_result.encode("utf-8"))


    def __repr__(self, **extra_params) -> str:
        parameters = {
            "model": self.model,
            **extra_params,
            "last_result": self.last_result,
            "output_filename": self.output_filename
        }
        return (
            f"{self.__class__.__name__}(" +
            "; ".join(f"{k}='{v}'" for k, v in parameters.items())
            + ")"
        )
    

class Inference(ModelExecution):

    KEY = "Inf"

    def __init__(self, id: int, model: ModelConfig, prompt: Prompt):
        super().__init__(id, model)
        self._prompt = prompt

    @property
    def prompt(this) -> Prompt:
        return this._prompt
    
    def setup(self: Self) -> Pipeline:
        return pipeline(
            task="text-generation",
            model=self._model['local'],
            num_workers=4,
            framework="pt",
            torch_dtype="auto",
            trust_remote_code = False,
            device_map="auto",
            model_kwargs=self._model['params']
        )
    
    def execute(self, model: Pipeline) -> str:
        return model(str(self.prompt))[0]['generated_text']

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prompt={self.prompt}; "
            f"model={self.model})"
        )