from time import time
from abc import ABC, abstractmethod
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
        return this._output_filename or ""

    def run(self, single=True, explore_comb=False, gen_params=None, to_save: bool=True) -> Any|list[Any]:
        self._model.load_instance(**self.setup())
        instance = self._model.instance
        
        ## Get combinations from generation parameters. Uncheck this for training 
        combinations = list()
        if gen_params is None:
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

        logger.info(f"Running execution '{self.id}'...")
        logger.info(f"\t model = {self.modelcfg['name']} ;")
        logger.info(f"\t configuration = {self.modelcfg['gen_params']} ;")
        logger.info(f"\t running combinations = {len(combinations)};")
        
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

        if to_save:
            logger.info(f"Saving result in '{self.output_filename}'...")
            self.save(exec_time=dtime)

        if single:
            self._model.teardown()
        return self._last_results


    @abstractmethod
    def setup(self):
        raise NotImplementedError
    
    @abstractmethod
    def execute(self, model, gen_params: dict[str, Any]) -> Any:
        raise NotImplementedError
    
    def save(self: Self, exec_time: float):
        assert self.last_results is not None

        if not self.output_filename:
            for params, result in self.last_results:
                print((f'Params:{params}\nResult:"""\n{result}\n"""\n' if params else f'"""\n{result}\n"""\n'))
            return

        outputfile = get_actual_path(self.output_filename, mode="result")

        if outputfile.exists():
            outputfile.unlink()

        if not isinstance(self.last_results, list):
            if not isinstance(self.last_results, tuple) or len(self.last_results) != 2:
                self._last_results = (None, self.last_results)
            self._last_results = [self.last_results]

        with outputfile.open("ab") as o:
            for params, result in self.last_results:
                o.write((f'Params:{params}\nResult:"""\n{result}\n"""\n' if params else f'"""\n{result}\n"""\nExecTime:{exec_time:.2f}s').encode("utf-8"))



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
    
class ModelBatch:

    def __init__(self, model: ModelConfig, executions: set[ModelExecution]) -> None:
        self.model = model
        self.executions = executions

        for exec in executions:
            if exec.modelcfg != model:
                logger.debug(f"Execution '{exec}' is a outlier of this batch with model '{model['name']}'!")
                raise ValueError(f"Batch cannot hold execution '{exec.id}' as the model differs.")
        
        logger.debug(f"Batch for the model '{model['name']}' has {len(executions)} executions.")
        
    def run(self, benchmode=False):

        logger.info(f"Running {len(self.executions)} executions of the model '{self.model['name']}' as a batch...")

        if not benchmode:
            for exec in self.executions:
                try:
                    exec.run(single=False)
                except Exception as e:
                    logger.error(f"Executing {exec.id}: {e}")
                    continue
            self.model.teardown()
            return


        from src.evaluation import LLMBenchmark
        assert all(isinstance(b, LLMBenchmark) for b in self.executions)

        def score(config_id, params: dict[str, Any|list[Any]], param):
            configuration = {p: v[0] if isinstance(v, list) else v for p, v in params.items()}
            logger.info(f"[DYNAMIC] Testing configuration {configuration}  with {param} = {params[param]}")

            result = 0
            output_file = get_actual_path(f"benchmode{config_id}_{param}", mode="result")

            with output_file.open('a') as r:
                r.write(f"Configuration: {configuration} for {param}={params[param]}\n")

                for bench in self.executions:
                    _, metrics = bench.run(single=False, explore_comb=False, gen_params=configuration, to_save=False)
                    r.write(f"{metrics}\n")
                    result += bench.WEIGHT * bench.get_score(metrics)

                r.write(f"Configuration cmp score = {result}\n\n\n")
            self.model.teardown()
            logger.info(f"[DYNAMIC] Scored configuration {param}={params[param]} with score={result}")

            return result

        gen_params = self.model["gen_params"].copy()
        priority_queue = self.model["gen_priorities"]
        config_id = 1
        # best_score = score(config_id, gen_params, "max_new_tokens")
        best_score = 0
        config_id += 1

        logger.info(f"Benchmode as been activated! Priority queue: {priority_queue}")
        for param in priority_queue:
            # Discard param
            if param not in gen_params or not isinstance((comb :=gen_params[param]), list):
                continue
            
            if len(comb) == 1:
                gen_params[param] = comb[0]
                continue

            comb = comb[1:]

            best_value, value_score = max((v, score(config_id, {
                **gen_params,
                param: v
            }, param)) for v in comb)

            if value_score > best_score:
                best_score = value_score
                assigned_value = best_value
            else:
                assigned_value = comb[0]

            logger.info(f"[DYNAMIC] Assigning {param}={assigned_value}")
            gen_params[param] = assigned_value
            config_id += 1


        assert all(not isinstance(gen_params[k], list) for k in priority_queue)

        logger.info(f"Benchmode as ended in {config_id} configurations! Best config {gen_params}")
    
        self.model.teardown()

    @staticmethod
    def from_list(executions: list[ModelExecution]) -> list['ModelBatch']:
        model_executions: dict[ModelConfig, list[ModelExecution]] = {}
        for exec in executions:
            if exec.modelcfg not in model_executions:
                model_executions[exec.modelcfg] = []

            model_executions[exec.modelcfg].append(exec)

        for model in model_executions.copy(): # keeping references
            model_executions[model].sort(key=lambda e: e.id)

        return list(ModelBatch(model, executions) for model, executions in model_executions.items())

