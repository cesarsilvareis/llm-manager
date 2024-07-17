import pandas as pd, evaluate

from typing import Literal, Any
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from src.evaluation import LLMBenchmark, INDEX_COLUMN
from src.model import ModelConfig
from src.utils import adapt_prompt

class PubMedSummary(LLMBenchmark):

  REPEAT = 1
  WEIGHT = .3

  def __init__(self, id: int, 
               modelcfg: ModelConfig,
               outputfile: str="",
               save_latents: Literal["output", "essential", "metric"]="metric") -> None:
    super().__init__(id, modelcfg, outputfile,  save_latents)
    self.metric = evaluate.load("rouge", trust_remote_code=True)


  def load_data(self, *args, **kwargs) -> Dataset:
    dataset = load_dataset("ccdv/pubmed-summarization", "section", split="test", *args, **kwargs)
    return dataset.select(range(55))
  
  def select_data(self):
    return self.dataset.shuffle(seed=7 * self.REPEAT).select(range(25))

  def define_prompt(self, medical_document):
    return [
      {"role": "system", "content": adapt_prompt(f"""\
        You are a medical researcher revising the current cutting-edge literature. In front of you is an interest article to see.
        Read carefully the user's document and propose a summarized abstract with essential ideas and findings of the paper.
        Just output the abstract with 100-200 words. Note that the user's article might be truncated.
        """
       )},
      {"role": "user", "content": f'Article:\n"""{medical_document}\n"""'}
    ]

  def tokenize(self, tokenizer: PreTrainedTokenizer, prompt, examples) -> BatchEncoding:
    prompts = [
      tokenizer.apply_chat_template(prompt(
          tokenizer.decode(tokenizer(article, truncation=True, max_length=800).input_ids, skip_special_tokens=True)
        ), tokenize=False, add_generation_prompt=True) for article in examples["article"]
    ]
    return tokenizer(prompts, padding=True, return_tensors="pt", return_attention_mask=True)

  def batching(self, tokenized_dataset: Dataset) -> DataLoader:
    return DataLoader(tokenized_dataset, shuffle=True, batch_size=5)

  def result_buffer(self) -> pd.DataFrame:
    return pd.DataFrame(columns=[INDEX_COLUMN, "prompt", "summary"])
  
  def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
    # Join required columns
    original_df = self.dataset.select_columns(column_names=["abstract"]
                              ).to_pandas()
    
    results["summary"] = results["summary"].str.replace("Abstract:", "").str.strip()

    metric_df = results.join(original_df, on=INDEX_COLUMN)

    if self._latent_mode == "essential":
      self.save_latent_data(metric_df)
    
    for idx in metric_df.index:
     
      summary = metric_df.loc[idx, "summary"]
      abstract = metric_df.loc[idx, "abstract"]

      score = self.metric.compute(
        predictions=[summary],
        references=[abstract],
        rouge_types=["rougeLsum"]
      )['rougeLsum']

      metric_df.loc[idx, "rouge"] = score

    if self._latent_mode == "metric":
      self.save_latent_data(metric_df)

    return metric_df.iloc[:, -1:]
  
  def get_score(self, results: pd.DataFrame) -> float:
    return results.at["mean", "rouge"]
