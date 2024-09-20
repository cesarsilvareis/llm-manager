import re
from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict
from operator import itemgetter

from src.tasks import ED_MCQ
from src.training.data_preparation import EDStep2

class EDStep3(EDStep2): # Similar Questions are removable

  def __init__(self, task: ED_MCQ, 
               dataset: Dataset|DatasetDict, # dataset 2
               cos_similarity_threshold: float = 0.9,
               bleurt_similarity_threshold: float = 1.025,
               split: str|None=None
               ) -> None:
    super().__init__(task, dataset, split)

    self._cos_similarity_threshold = cos_similarity_threshold
    self._bleurt_similarity_threshold = bleurt_similarity_threshold

  def initialize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    from evaluate import load
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine

    scorer = load("bleurt", "bleurt-tiny-128")
    embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    seen_qa_pairs = list()
    seen_embeedings = list()
    def is_unique_inside(record):
      explore_questions = list(set(q for q in record[self._task.QUESTION_LABEL] if q not in seen_qa_pairs))
      res = { q: True for q in record[self._task.QUESTION_LABEL] }
      
      if len(seen_qa_pairs) == 0:
        first_question = explore_questions.pop(0)
        seen_qa_pairs.append(first_question)
        seen_embeedings.append(embedding_model.encode(first_question, show_progress_bar=False))

      for current_question in explore_questions:

        emb_question = embedding_model.encode(current_question, show_progress_bar=False)
        
        cos_sim = max([1 - cosine(emb_question, emb) for emb in seen_embeedings])
        
        if cos_sim < self._cos_similarity_threshold:
          
          score = max(scorer.compute(references=seen_qa_pairs[::32], predictions=[current_question]*len(seen_qa_pairs[::32]))['scores'])
      
          if score < self._bleurt_similarity_threshold:
            seen_qa_pairs.append(current_question)
            seen_embeedings.append(emb_question)
            continue

        res[current_question] = False

      return list(res[q] for q in record[self._task.QUESTION_LABEL])
    

    return dataset.shuffle(self._task.SEED).filter(is_unique_inside, batch_size=512, batched=True, load_from_cache_file=False)

  def fn_process(self, examples, _) -> Dataset:
    return super().fn_process(examples, _)

  def finalize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    return super().finalize(dataset, split)

