import re
from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict
from operator import itemgetter

from src.tasks import ED_MCQ
from src.training.data_preparation import PreProcessing

class EDStep2(PreProcessing): # MCQ -> QA pairs

  def __init__(self, task: ED_MCQ, 
               dataset: Dataset|DatasetDict,  # dataset 1
               split: str|None=None,
               ) -> None:
    super().__init__(task, dataset, split)

    self._relevant_columns = [
      self._task.QUESTION_LABEL, self._task.ANSWER_LABEL, self._task.OUTPUT_LABEL, self._task.MEDICAL_SUBJECT_LABEL
    ]


  def initialize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    return dataset.sort(self._task.QUESTION_LABEL)

  def _separator(self, examples):
    assert all(lbl in examples for lbl in self._relevant_columns)

    return zip(*list(examples[c] for c in self._relevant_columns))
    

  def fn_process(self, examples, _) -> Dataset:

    qa_groups = list()
    for question, answer, label, subj in self._separator(examples):
      if any((question, _, label, _) == (q, _, l, _) for q, _, l, _ in qa_groups[-5:]):
        continue

      qa_groups.append((question, answer, label, subj))
      
    questions, answers, labels, subjs = zip(*qa_groups)

    return { lbl: data for lbl, data in zip(
      (self._task.QUESTION_LABEL, self._task.ANSWER_LABEL, self._task.OUTPUT_LABEL, self._task.MEDICAL_SUBJECT_LABEL), 
      map(list, (questions, answers, labels, subjs))
    )}

  def finalize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    return dataset.select_columns(self._relevant_columns)
    

