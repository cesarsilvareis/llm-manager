from time import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self, Optional
from src import get_actual_path
from src.logger import get_logger
from src.model import ModelConfig
from typing import Any
from itertools import product
from src.utils import return_all

logger = get_logger(__name__)

class ModelExecution(ABC):

    ACTIVIVE_EXECUTIONS: set[int]=set()

    def __init__(self, id: int, model: ModelConfig, output_filename: str):
        assert id not in self.ACTIVIVE_EXECUTIONS
        ModelExecution.ACTIVIVE_EXECUTIONS.add(id)
        
        self._id = id
        self._model = model
        self._last_results = None
        self._output_filename = output_filename

    @property
    def id(this) -> int:
        return this._id
    
    @property
    def modelcfg(this) -> ModelConfig:
        return this._model
    
    @property
    def last_results(this) -> Optional[Any|list[Any]]:
        return this._last_results
    
    @property
    def output_filename(this) -> str:
        return this._output_filename

    def run(self, single=True, explore_comb=False) -> str:
        self._model.load_instance(**self.setup())
        instance = self._model.instance
        
        
        ## Get combinations from generation parameters. Uncheck this for training 
        combinations = list()
        gen_params: dict[str, Any] = self.modelcfg["gen_params"] # this is a copy
        if explore_comb:
            for param, value in gen_params.items():
                if isinstance(value, list): # i.e., there is no change for the param
                    continue
                gen_params[param] = [value]

            param_comb = product(*gen_params.values())
            for comb in param_comb:
                combinations.append(dict(zip(gen_params.keys(), comb)))
        else:
            for param, value in gen_params.items():
                if not isinstance(value, list):
                    continue
                gen_params[param] = value[0] # choosing the first element
            
            combinations.append(gen_params)

        assert len(combinations) > 0

        logger.info(f"Running execution '{self.id}' {len(combinations)} times (rel., combinations)...")
        stime = time()
        try:
            self._last_results = return_all(
                self.execute, "gen_params", 
                *instance if isinstance(instance, tuple) else [instance], # .. what?... pythonio? :) 
                gen_params=combinations,
                return_as_tuple=True
            )
        except Exception as e:
            logger.error(f"Error in executing model {self.modelcfg['name']}: {e}")
            raise e
            logger.info(f"Finalizing this execution '{self.id}' without having a result...")
            return

        dtime = time() - stime
        logger.info(f"Execution '{self.id}' ran for {int(dtime)}s")

        logger.info(f"Saving result in '{self.output_filename}'...")
        self.save()

        if single:
            self._model.teardown()


    @abstractmethod
    def setup(self):
        raise NotImplementedError
    
    @abstractmethod
    def execute(self, model, gen_params: dict[str, Any]) -> Any:
        raise NotImplementedError
    
    def save(self: Self):
        assert self.last_results is not None

        outputfile = get_actual_path(self.output_filename, mode="result")

        if outputfile.exists():
            outputfile.unlink()

        if not isinstance(self.last_results, list):
            if not isinstance(self.last_results, tuple) or len(self.last_results) != 2:
                self._last_results = (None, self.last_results)
            self._last_results = [self.last_results]

        with outputfile.open("ab") as o:
            for params, result in self.last_results:
                o.write((f'Params:{params}\nResult:"""\n{result}\n"""\n' if params else f'"""\n{result}\n"""\n').encode("utf-8"))



    def __repr__(self, **extra_params) -> str:
        parameters = {
            "id": self.id,
            "model": self.modelcfg["name"],
            **extra_params,
            "last_result": self.last_results,
            "output_filename": self.output_filename
        }
        return (
            f"{self.__class__.__name__}(" +
            "; ".join(f"{k}={repr(v)}" for k, v in parameters.items())
            + ")"
        )
    
class ModelBatch(Sequence):

    def __init__(self, model: ModelConfig, executions: set[ModelExecution]) -> None:
        self.model = model
        self.executions = executions

        for exec in executions:
            if exec.modelcfg != model:
                logger.debug(f"Execution '{exec}' is a outlier of this batch with model '{model['name']}'!")
                raise ValueError(f"Batch cannot hold execution '{exec.id}' as the model differs.")
        
        logger.debug(f"Batch for the model '{model['name']}' has {len(executions)} executions.")
        
    def run(self):

        logger.info(f"Running {len(self.executions)} executions of the model '{self.model['name']}'...")

        for exec in self.executions:
            exec.run(single=False)
    
        self.model.teardown()

    @staticmethod
    def from_list(executions: list[ModelExecution]) -> list['ModelBatch']:
        model_executions: dict[ModelConfig, list[ModelExecution]] = {}
        for exec in executions:
            if exec.modelcfg not in model_executions:
                model_executions[exec.modelcfg] = []

            model_executions[exec.modelcfg].append(exec)

        for model in model_executions.copy(): # keeping references
            model_executions[model] = model_executions[model].sort(key=lambda e: e.id)

        return list(ModelBatch(model, executions) for model, executions in model_executions)

