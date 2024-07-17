import pandas as pd, evaluate

from typing import Literal, Any
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from src.evaluation import LLMBenchmark, INDEX_COLUMN
from src.model import ModelConfig
from src.utils import adapt_prompt

class ClinicalParaph(LLMBenchmark):

  REPEAT = 1
  WEIGHT = .4

  WEIGHTED_SCORE = lambda v, r: .4*v + .6*((r + 1) / 2) # with normalization

  def __init__(self, id: int, 
               modelcfg: ModelConfig,
               outputfile: str="",
               save_latents: Literal["output", "essential", "metric"]="metric") -> None:
    super().__init__(id, modelcfg, outputfile,  save_latents)
    # Load metric from HF
    self.nonvariance = evaluate.load("bleu", trust_remote_code=True)
    self.retention = evaluate.load("bleurt", trust_remote_code=True)



  def load_data(self, *args, **kwargs) -> Dataset:
    dataset = load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes", split="train", *args, **kwargs)
    return dataset.filter(lambda e: e["task"] == "Paraphrasing") \
                  .select(range(75))

  def select_data(self):
    return self.dataset.shuffle(seed=7 * self.REPEAT).select(range(75))

  def define_prompt(self, discharge, question, attempt):
    return [
      {"role": "system", "content": adapt_prompt(f"""\
        You are given with the follow discharge summary of a patient:
        <condiction>
        {discharge}
        </condition>
        
        Your task is to rephrase the user's answer to next question. While trying to keep nearly the number of used words, adapt this text to a new version where you utilize medical language suited to the complexity of medical academy.
        <question>
        {question}
        </question>
        
        Just respond to the user just with your paraphrase.
        """
       )},
      {"role": "user", "content": f"{attempt}"}
    ]

  def tokenize(self, tokenizer: PreTrainedTokenizer, prompt, examples) -> BatchEncoding:
    prompts = [
      tokenizer.apply_chat_template(prompt(
        tokenizer.decode(tokenizer(discharge, truncation=True, max_length=800).input_ids, skip_special_tokens=True),
          question, attempt), tokenize=False, add_generation_prompt=True)
        for discharge, question, attempt in zip(examples["note"], examples["question"], examples["answer"])
    ]
    return tokenizer(prompts, padding=True, return_tensors="pt", return_attention_mask=True)

  def batching(self, tokenized_dataset: Dataset) -> DataLoader:
    return DataLoader(tokenized_dataset, shuffle=True, batch_size=10)

  def result_buffer(self) -> pd.DataFrame:
    return pd.DataFrame(columns=[INDEX_COLUMN, "prompt", "paraphrase"])
  
  def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
    # Join required columns
    original_df = self.dataset.select_columns(column_names=["answer"] # chatgpt attempt (simple)
                              ).to_pandas()

    metric_df = results.join(original_df, on=INDEX_COLUMN)

    if self._latent_mode == "essential":
      self.save_latent_data(metric_df)
    
    for idx in metric_df.index:
     
      paraphrase = metric_df.loc[idx, "paraphrase"]
      previous = metric_df.loc[idx, "answer"]

      bleu_score = self.nonvariance.compute( # lower the better
        predictions=[paraphrase],
        references=[previous],
      )['precisions'][0]

      bleurt_score = self.retention.compute(
        predictions=[paraphrase],
        references=[previous]
      )["scores"][0]

      metric_df.loc[idx, "variance"] = 1 - bleu_score
      metric_df.loc[idx, "retention"] = bleurt_score
      metric_df.loc[idx, "weighted_mean"] = ClinicalParaph.WEIGHTED_SCORE(1-bleu_score, bleurt_score)

    if self._latent_mode == "metric":
      self.save_latent_data(metric_df)

    return metric_df.iloc[:, -3:]
  
  def get_score(self, results: pd.DataFrame) -> float:
    return results.at["mean", "weighted_mean"]