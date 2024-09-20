from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict

from src.tasks import Task

class PreProcessing(ABC):

  def __init__(self, task: Task, dataset: Dataset|DatasetDict, split: str|None=None) -> None:
    self._task = task

    assert isinstance(dataset, DatasetDict) or split is not None

    self._dataset = dataset
    self._split = split

  @abstractmethod
  def initialize(self, dataset: Dataset, split: str|None=None) -> Dataset:
    ...   

  @abstractmethod
  def fn_process(self, examples, idx) -> Dataset:
    ...

  @abstractmethod
  def finalize(self, dataset: Dataset, split: str) -> Dataset:
    ...

  def _process(self, dataset: Dataset, split: str) -> Dataset:

    dataset = self.initialize(dataset, split)

    dataset = dataset.map(self.fn_process, with_indices=True, batched=True, batch_size=12, 
        load_from_cache_file=False, keep_in_memory=True, remove_columns=dataset.column_names
    )

    print(split, dataset)
    return self.finalize(dataset, split)

  def run(self) -> Dataset|DatasetDict:
    if isinstance(self._dataset, DatasetDict):
      return DatasetDict({
        split: self._process(self._dataset[split], split)
        for split in self._dataset.column_names
      })

    return self._process(self._dataset, self._split)

  