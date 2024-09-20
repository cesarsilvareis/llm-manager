import re
from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict
from operator import itemgetter

from src.tasks import ED_MCQ
from src.training.data_preparation import PreProcessing

class EDStep1(PreProcessing): # MCQ -> QA pairs

  def __init__(self, task: ED_MCQ, 
               dataset: Dataset|DatasetDict,
               question_lbl: str,
               options_lbl: list[str],
               true_lbl: str,
               medical_subject_lbl: str,
               initial_sizes: int|list[int]|None,
               split: str|None=None
               ) -> None:
    super().__init__(task, dataset, split)

    self._question_lbl = question_lbl
    self._options_lbl = options_lbl
    self._true_lbl = true_lbl
    self._medical_subject_lbl = medical_subject_lbl

    self._relevant_columns = {self._question_lbl, *self._options_lbl, self._true_lbl, self._medical_subject_lbl}

    self._initial_sizes = initial_sizes

  def initialize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    match split:
      case "train"|"validation":
        dataset = dataset.filter(lambda x: x["choice_type"] == "single")
      case "test":
        mcq_pattern = r"Q:(.*)\?\s*\{'A':\s*['\"](.*)['\"], 'B':\s*['\"](.*)['\"], 'C':\s*['\"](.*)['\"], 'D':\s*['\"](.*)['\"], 'E':\s*['\"](.*)['\"]\}"
        cop_pattern = r"([A-Z]): .*"
        mcq_extractor, cop_extractor = re.compile(mcq_pattern, re.DOTALL), re.compile(cop_pattern, re.DOTALL)

        dataset = dataset.rename_columns({
          "input": self._question_lbl,
          "output": self._true_lbl
        })

        dataset = dataset.filter(lambda x: any("endocarditis" in c for c in [x[self._question_lbl], x[self._true_lbl]]), batched=False)

        dataset = dataset.map(lambda x: {
            **{ l: d for l, d in zip((self._question_lbl, *self._options_lbl), mcq_extractor.search(x[self._question_lbl]).groups())
            }, self._true_lbl: ord(cop_extractor.search(x[self._true_lbl]).group(1).strip().lower()) - ord("a")
        }, batched=False)

    return dataset.shuffle(seed=self._task.SEED)\
                  .select(range((self._initial_sizes[split] if split else self._initial_sizes) or dataset.num_rows))

  def _separator(self, examples):
    assert all(lbl in examples for lbl in self._relevant_columns)

    return zip(
      examples[self._question_lbl],
      *itemgetter(*self._options_lbl)(examples),
      examples[self._true_lbl],
      examples[self._medical_subject_lbl]
    )

  def fn_process(self, examples, _) -> Dataset:

    qa_groups = list()
    for question, *options, label, subj in self._separator(examples):
      qa_groups.extend(
        (question, opt, self._task.get_label(i, label, return_id=True)[0], subj) 
          for i, opt in enumerate(options)
      )
      
    questions, answers, labels, subjs = zip(*qa_groups)

    return { lbl: data for lbl, data in zip(
      (self._task.QUESTION_LABEL, self._task.ANSWER_LABEL, self._task.OUTPUT_LABEL, self._task.MEDICAL_SUBJECT_LABEL),
      map(list, (questions, answers, labels, subjs))
    )}


  def finalize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    if split != "validation":
      return dataset.select_columns((self._task.QUESTION_LABEL, self._task.ANSWER_LABEL, self._task.OUTPUT_LABEL))
    
    return dataset.select_columns((self._task.QUESTION_LABEL, self._task.ANSWER_LABEL, self._task.OUTPUT_LABEL, self._task.MEDICAL_SUBJECT_LABEL))
    

