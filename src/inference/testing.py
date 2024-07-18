import numpy as np
from src import get_actual_path
from src.execution import ModelExecution
from src.model import ModelConfig
from src.tasks import Task, TaskType
from typing import Self, Any
from transformers import Trainer, AutoTokenizer
from datasets import load_from_disk, Dataset, DatasetDict


class Testing(ModelExecution):

    TEST_DATA = "test"

    def __init__(self, id: int, modelcfg: ModelConfig, task: Task|TaskType, data_local: Dataset|DatasetDict, output_filename: str):
        super().__init__(id, modelcfg, output_filename)

        self._task = task
        if isinstance(task, TaskType):
            self._task = Task(task)
       
        self._dataset = data_local
        if isinstance(self._dataset, DatasetDict):
            self._dataset = self._dataset[Testing.TEST_DATA]

    @property
    def test(this) -> Dataset:
        return this._dataset

    def setup(self: Self):
        def prepare_trainer(checkpoint_local, model_kwargs) -> Trainer:
            print(model_kwargs)
            pretrained_model = self._task.get_pretrained_model_forme(
                checkpoint_local, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_local)

            return Trainer(
                model=pretrained_model,
                tokenizer=tokenizer,
                compute_metrics=self._task.compute_metrics
            )

        return {
            "caller": prepare_trainer,
            "checkpoint_local": get_actual_path(self.modelcfg.local, mode="model"),
            "model_kwargs": self.modelcfg["model_params"],
        }

    def execute(self, trainer: Trainer, **_) -> Any|list[Any]:
        tokenized_dataset = self._task._trainer_scheme.tokenizing_data(self.test, self._task, trainer.tokenizer)
        test_predictions = trainer.predict(tokenized_dataset)

        return {
            "metrics": test_predictions.metrics,
            "prediction": np.argmax(test_predictions.predictions, axis=-1), 
            "ground_truth": test_predictions.label_ids,
            **trainer.evaluate(tokenized_dataset),
        }


    def __repr__(self, **_) -> str:
        test_set = self.test
        return super().__repr__(test=test_set[slice(.1*test_set.num_rows)])
