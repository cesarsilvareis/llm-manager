from src import get_actual_path
from src.execution import ModelExecution
from src.model import ModelConfig
from typing import Self, Any
from transformers import Trainer, AutoModelForSequenceClassification
from datasets import load_from_disk, Dataset, DatasetDict


class Testing(ModelExecution):

    TEST_DATA = "test"

    def __init__(self, id: int, modelcfg: ModelConfig, data_local: str, output_filename: str):
        super().__init__(id, modelcfg, output_filename)

        assert data_local

        self._dataset_local = get_actual_path(data_local, mode="data")
        self._dataset = load_from_disk(self._dataset_local)
        if isinstance(self._dataset, DatasetDict):
            self._dataset = self._dataset[Testing.TEST_DATA]

    @property
    def test(this) -> Dataset:
        return this._dataset

    def setup(self: Self):
        def prepare_trainer(checkpoint_local, ) -> Trainer:
            trained_model = AutoModelForSequenceClassification(checkpoint_local)

        return {
            "caller": prepare_trainer,
            "task": "text-generation",
            "model": get_actual_path(self.modelcfg.local, mode="model"),
            "num_workers": 4,
            "torch_dtype": "auto",
            "trust_remote_code": False,
            "device_map": "cuda",
            "model_kwargs": self.modelcfg["model_params"],
        }

    def execute(self, trainer: Trainer) -> list[str]:
        trainer.tokenizer
        ...

    def __repr__(self, **_) -> str:
        test_set = self.test
        return super().__repr__(test=test_set[slice(.1*test_set.num_rows)])
