from abc import ABC, abstractmethod
from peft import TaskType
from transformers import (
  PreTrainedModel,
  PreTrainedTokenizer,
  AutoModelForCausalLM,
  AutoModelForSequenceClassification
)
from datasets import Dataset, DatasetDict
from evaluate import EvaluationModule

class Task(ABC):

  MODEL_FOR_TASK = {
    TaskType.CAUSAL_LM: AutoModelForCausalLM,
    TaskType.SEQ_CLS: AutoModelForSequenceClassification
    # ...
  }

  def __init__(self, type: TaskType, trainer_scheme) -> None:
    self._type = type
    self._trainer_scheme = trainer_scheme
    self._epoch = 0


  @property
  def type(this) -> TaskType:
    return this._type


  def get_pretrained_model_forme(self, *args, **kwargs) -> PreTrainedModel:
    return Task.MODEL_FOR_TASK[self.type].from_pretrained(*args, **self.get_parameters(), **kwargs)

  @abstractmethod
  def get_max_metric(self) -> EvaluationModule:
    ...

  @abstractmethod
  def get_parameters(self) -> dict:
    ...

  @abstractmethod
  def tokenization(examples, tokenizer: PreTrainedTokenizer, *args, **kwargs) -> Dataset|DatasetDict:
    ...

  @abstractmethod
  def compute_metrics(eval_preds) -> dict[str, float]:
    ...

  

  