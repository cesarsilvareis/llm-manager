import pandas as pd
from src import get_actual_path
from src.model import ModelConfig
from src.logger import get_logger
from src.execution import ModelExecution
from abc import abstractmethod
from typing import Literal, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from datasets import Dataset
from torch.utils.data import DataLoader

logger = get_logger(__name__)

INDEX_COLUMN = "index"

class LLMBenchmark(ModelExecution):

  REPEAT: int = 1
  WEIGHT: int

  def __init__(self, id: int, modelcfg: ModelConfig, output_filename: str="", 
               save_latents: Literal["ouputs", "essential", "metric"]="outputs") -> None:
    super().__init__(id, modelcfg, output_filename)
    self.cache_dir = get_actual_path(output_filename, mode="data")
    self.dataset = self.load_data(cache_dir=self.cache_dir)
    self.dataset = self.dataset.map(lambda e, i: {INDEX_COLUMN: i, **e}, with_indices=True)
    self._latent_mode = save_latents
    self.tokenized_dataset = None

  @abstractmethod
  def load_data(self, *args, **kwargs) -> Dataset:
    ...

  @abstractmethod
  def select_data(self) -> Dataset:
    ...

  def setup(self):
    # self.modelcfg.teardown()

    def loader(local, model_params):
      model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local,
        **model_params,
      )

      model = model.eval()

      tokenizer = AutoTokenizer.from_pretrained(local, use_fast=False)

      # For decoders only
      tokenizer.padding_side = "left"
      tokenizer.pad_token = tokenizer.eos_token
      model.resize_token_embeddings(len(tokenizer))
      model.config.pad_token_id = model.config.eos_token_id
      
      logger.info(f"Loaded model '{self.modelcfg['name']}' with  footprint: {(model.get_memory_footprint()/1e9):.3f} GB of (V)RAM")
      return model, tokenizer

    return {
      "caller": loader,
      "local": get_actual_path(self.modelcfg.local, "model"),
      "model_params": self.modelcfg["model_params"]
    }
  

  def save_latent_data(self, latent_data: pd.DataFrame):
    output_file = get_actual_path(f"{self.output_filename}.csv", mode="data")
    logger.info(f"[EXEC {self.id}] Saving latent {self._latent_mode}s. Results on size {latent_data.shape} in {output_file}...")
    latent_data.to_csv(output_file)

  @abstractmethod
  def define_prompt(self, *args, **kwargs):
    ...

  @abstractmethod
  def tokenize(self, tokenizer: PreTrainedTokenizer, prompt, examples) -> BatchEncoding:
    ...

  def tokenization(self, examples, tokenizer: PreTrainedTokenizer, prompt):
    tokenized = self.tokenize(tokenizer, prompt, examples)
    tokenized[INDEX_COLUMN] = examples[INDEX_COLUMN]
    return tokenized

  @abstractmethod
  def batching(self, tokenized_dataset: Dataset, tokenizer: PreTrainedTokenizer) -> DataLoader:
    ...

  @abstractmethod
  def result_buffer(self) -> pd.DataFrame:
    ...

  @abstractmethod
  def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
    ...

  @abstractmethod
  def get_score(self, results) -> float:
    ...


  def execute(self, model, tokenizer: PreTrainedTokenizer, gen_params: dict[str, Any]) -> Any | list[Any]:
    results = []
    for rep in range(1, self.REPEAT + 1):
      # exec_dataset = self.select_data()
      # logger.debug(f"Executing the '{rep}' time(s), this with dataset {exec_dataset}")
      
      result = self._execute(model, tokenizer, gen_params)
      # logger.info(f"Result of {self.__class__.__name__} (rep={rep}): {result}")
      results.append(result)

    # Combine the DataFrames
    combined_df = pd.concat(results, axis=1)

    # Compute the mean of the combined DataFrame
    return combined_df.T.groupby(combined_df.columns).mean()


  def _execute(self, model, tokenizer: PreTrainedTokenizer, gen_params: dict[str, Any]) -> Any|list[Any]:
    if self.tokenized_dataset is None:
      self.tokenized_dataset = self.dataset.map(self.tokenization, new_fingerprint=self.output_filename,
          fn_kwargs={"tokenizer": tokenizer, "prompt": self.define_prompt}, batched=True, 
          remove_columns=self.dataset.column_names, load_from_cache_file=False)
  
      self.tokenized_dataset.set_format(type="torch", columns=self.tokenized_dataset.column_names)

    logger.debug((
      f"Dataset tokenization: {repr(self.tokenized_dataset)}\n" 
      f"Cache files:\n"
      f"\toriginal: {self.dataset.cache_files} (should exist)\n"
      f"\ttokenized: {self.tokenized_dataset.cache_files} (should be empty)")
    )

    logger.info(f"Finished tokenization! Starting inference of {self.tokenized_dataset.num_rows} prompts...")

    terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    batches = self.batching(self.tokenized_dataset, tokenizer)

    results_df = self.result_buffer()

    import torch
    with torch.no_grad():
      batch_size = batches.batch_size
      for i, batch in enumerate(batches):
        idx = batch[INDEX_COLUMN]
        input_ids = batch["input_ids"].to(device=model.device)
        attention_mask = batch["attention_mask"].to(device=model.device)
        
        logger.info(f"\tInfering batch {i+1}/{len(batches)} of {len(input_ids)} size...")
        outputs_ids = model.generate(input_ids, attention_mask=attention_mask, num_return_sequences=1, eos_token_id=terminators, **gen_params)

        for j, (idx_tensor, input_enc, output_enc) in enumerate(zip(idx, input_ids, outputs_ids)):
          idx = idx_tensor.item()
          input = tokenizer.decode(input_enc, skip_special_tokens=True)
          output = tokenizer.decode(output_enc[input_enc.shape[-1]:], skip_special_tokens=True)
          results_df.loc[i * batch_size + j, results_df.columns] = idx, input, output
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    results_df.set_index(INDEX_COLUMN, inplace=True)
    logger.info("Inference has finished! Starting computing metrics...")

    if self._latent_mode == "output":
      self.save_latent_data(results_df)
      return ""

    metrics_df = self.compute_metrics(results_df)
    metrics = metrics_df.agg(['mean', 'median']).T

    return metrics
  
