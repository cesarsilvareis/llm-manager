from abc import ABC, abstractmethod
from peft import TaskType
from transformers import (
  PreTrainedModel,
  PreTrainedTokenizer,
  AutoModelForCausalLM,
  AutoModelForSequenceClassification
)
from datasets import Dataset, DatasetDict

class Task(ABC):

  MODEL_FOR_TASK = {
    TaskType.CAUSAL_LM: AutoModelForCausalLM,
    TaskType.SEQ_CLS: AutoModelForSequenceClassification
    # ...
  }

  def __init__(self, type: TaskType, trainer_scheme) -> None:
    self._type = type
    self._trainer_scheme = trainer_scheme


  @property
  def type(this) -> TaskType:
    return this._type


  def get_pretrained_model_forme(self, *args, **kwargs) -> PreTrainedModel:
    return Task.MODEL_FOR_TASK[self.type].from_pretrained(*args, **self.get_parameters(), **kwargs)


  @abstractmethod
  def get_parameters(self) -> dict:
    ...

  @abstractmethod
  def tokenization(examples, tokenizer: PreTrainedTokenizer, *args, **kwargs) -> Dataset|DatasetDict:
    ...

  @abstractmethod
  def compute_metrics(eval_preds) -> dict[str, float]:
    ...

  

  