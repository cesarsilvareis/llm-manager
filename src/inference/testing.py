import numpy as np
from jinja2 import Template
from src import get_actual_path
from src.utils import adapt_prompt
from src.execution import ModelExecution
from src.model import ModelConfig
from src.tasks import Task, TaskType
from src.logger import get_logger
from typing import Self, Any
from transformers import Trainer, AutoTokenizer
from datasets import Dataset, DatasetDict

logger = get_logger(__name__)

class Testing(ModelExecution):

    TEST_DATA = "test"
    PRESENT_LINES = slice(14)
    REPEAT = 1

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
            pretrained_model.eval()

            # Configurations for our decoder-only transformer
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            pretrained_model.resize_token_embeddings(len(tokenizer))
            pretrained_model.config.pad_token_id = pretrained_model.config.eos_token_id

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

    def _execute(self, trainer: Trainer, **_) -> Any|list[Any]:
        tokenized_dataset = self._task._trainer_scheme.tokenizing_data(self.test, self._task, trainer.tokenizer)
        test_outputs = trainer.predict(tokenized_dataset)
        predictions = np.argmax(test_outputs.predictions, axis=-1)
        return {
            "predictions": predictions[Testing.PRESENT_LINES],
            "ground_truth": test_outputs.label_ids[Testing.PRESENT_LINES],
            "fp": np.sum((predictions == 1) & (test_outputs.label_ids == 0)), # = 0, max precision --> complete
            "fn": np.sum((predictions == 0) & (test_outputs.label_ids == 1)), # = 0, max recall --> sound
            "metrics": test_outputs.metrics,

        }
    
    def execute(self, trainer: Trainer, **_) -> Any|list[Any]:
        results = []
        logger.info(f"Testing a set of {self.test.num_rows} test cases (distr: {self.test.to_pandas()['label'].value_counts()})")
        for _ in range(Testing.REPEAT):
            results.append(self._execute(trainer))

        max_metric = self._task.get_max_metric().name
        return Template(adapt_prompt("""

            ==== MAIN METRICS ====

            Ground Truth:
            {{ results[0]["ground_truth"] }}

            Predictions:
            {% for r in results %}
            - {{ r["predictions"] }}

            FP: {{ r["fp"] }}
            FN: {{ r["fn"] }}

            {% endfor %}
            

            Final Score:
            {{ max_metric }}: {{ final_score }}
            
            ==== OTHER METRICS ====

            {% for r in results %}
            ; {{ r }}: {{ results[r] }}
            {% endfor %}

            """
        )).render(results=results, max_metric=max_metric, 
                  final_score=np.mean(list(r["metrics"][f"test_{max_metric}"] for r in results)))

    def __repr__(self, **_) -> str:
        test_set = self.test
        return super().__repr__(test=test_set[slice(.1*test_set.num_rows)])
