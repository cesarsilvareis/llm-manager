import pandas as pd, evaluate

from typing import Literal
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader

from src.evaluation import LLMBenchmark, INDEX_COLUMN
from src.model import ModelConfig
from src.utils import adapt_prompt

class PubMedSummary(LLMBenchmark):

  def __init__(self, id: int, 
               modelcfg: ModelConfig,
               save_latents: Literal["output", "essential", "metric"]="metric") -> None:
    super().__init__(id, modelcfg, "pubmedsmr", save_latents)


  def define_prompt(self, medical_document):
    return [
      {"role": "system", "content": adapt_prompt(f"""\
        You are a medical researcher revising the current cutting-edge literature. In front of you is an interest article to see.
        Read carefully the user's document and propose a summarized abstract with essential ideas and findings of the paper.
        Just output the abstract with the usual number of words."""
       )},
      {"role": "user", "content": f"Article:\n{medical_document}"}
    ]

  def tokenize(self, tokenizer: PreTrainedTokenizer, prompt, examples) -> BatchEncoding:
    prompts = [tokenizer.apply_chat_template(prompt(question), tokenize=False, add_generation_prompt=True) for question in examples["article"]]
    return tokenizer(prompts, padding=True, return_tensors="pt", return_attention_mask=True)

  def batching(self, tokenized_dataset: Dataset) -> DataLoader:
    return DataLoader(tokenized_dataset, shuffle=True, batch_size=10)

  def result_buffer(self) -> pd.DataFrame:
    return pd.DataFrame(columns=[INDEX_COLUMN, "article", "summary"])
  
  def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
    # Load metric from HF
    metric = evaluate.load("rouge", cache_dir=self.cache_dir, trust_remote_code=True)

    # Join required columns
    original_df = self.dataset.select_columns(column_names=["abstract"]
                              ).to_pandas()

    metric_df = results.join(original_df, on=INDEX_COLUMN)

    if self._latent_mode == "essential":
      self.save_latent_data(metric_df)
    
    for idx in metric_df.index:
     
      summary = metric_df.loc[idx, "summary"]
      abstract = metric_df.loc[idx, "abstract"]

      score = metric.compute(
        predictions=[summary],
        references=[abstract],
        rouge_types=["rougeLsum"]
      )['rougeLsum']

      
      col_name = f'{self.modelcfg["name"]} {"rougeLsum"} ({{0}})'
      metric_df.loc[idx, col_name.format("max")] = max(score)
      metric_df.loc[idx, col_name.format("diff")] = max(score) - max(score)
      metric_df.loc[idx, col_name.format("acc")] = int(max(score) > max(score))

    if self._latent_mode == "metric":
      self.save_latent_data(metric_df)

    return metric_df.iloc[:, -3:]