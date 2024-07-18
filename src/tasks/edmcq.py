import numpy as np
from src.tasks import Task, TaskType, Dataset, DatasetDict, PreTrainedTokenizer
from collections import ChainMap
from evaluate import EvaluationModule, load


class ED_MCQ(Task):

  SEED = 7

  CORRECT_LABEL = "Correct"
  INCORRECT_LABEL = "Incorrect"

  METRICS = [
    load("accuracy"), # the upper the chosen for scoring checkpoints
    load("recall"),
    load("precision"),
    load("f1")
  ]


  def __init__(self) -> None:
    from src.training import Training_EDMCQ
    super().__init__(TaskType.SEQ_CLS, trainer_scheme=Training_EDMCQ)

  @property
  def labels2ids(this):
    return {
      ED_MCQ.CORRECT_LABEL:   0,  # default case (False)
      ED_MCQ.INCORRECT_LABEL: 1   # alarming case (True)
    }
  
  @property
  def ids2labels(this):
    return { i : l for l, i in this.labels2ids.items() }
  
  @property
  def num_labels(this):
    return len(this.labels2ids)
  
  @property
  def max_metric(this) -> EvaluationModule:
    return this.METRICS[0]

  def get_label(self, id_opi: int, id_opc: int, return_id: bool=False) -> str:
    label = ED_MCQ.CORRECT_LABEL if id_opi == id_opc else ED_MCQ.INCORRECT_LABEL
    return (self.labels2ids[label], label) if return_id else label


  def get_parameters(self) -> dict:
    return { "label2id" : self.labels2ids, "id2label": self.ids2labels }

  def tokenization(self, examples, tokenizer: PreTrainedTokenizer, 
      question: str, answer: str, label: str, ctx_len: int=252) -> Dataset | DatasetDict:
    
    return { "labels": examples[label], **tokenizer(
      examples[question], examples[answer], # QA pairwise input
      truncation=True, max_length=ctx_len, return_tensors="np", return_overflowing_tokens=False)
    }
  

  def compute_metrics(self, eval_preds) -> dict[str, float]:
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1) 
    return dict(ChainMap(*(metric.compute(predictions=predictions, references=labels) 
                           for metric in ED_MCQ.METRICS)))

