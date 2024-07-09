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

  def __init__(self, id: int, modelcfg: ModelConfig, benchname: str, save_latents: Literal["ouputs", "essential", "metric"]="outputs") -> None:
    super().__init__(id, model=modelcfg, output_filename=benchname)
    self.cache_dir = get_actual_path(benchname, mode="data")
    self.dataset = self.load_data(cache_dir=self.cache_dir)
    self.dataset = self.dataset.map(lambda e, i: {INDEX_COLUMN: i, **e}, with_indices=True)
    self.benchname = benchname
    self._latent_mode = save_latents

  @abstractmethod
  def load_data(self, *args, **kwargs) -> Dataset:
    ...

  def setup(self):
    self.modelcfg.teardown()

    def loader(local, model_params):
      import torch
      model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_params 
      )

      model.eval()

      tokenizer = AutoTokenizer.from_pretrained(local)

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
    logger.info(f"[EXEC {self.id}] Saving latent {self._latent_mode}s...")
    latent_data.to_csv(get_actual_path(f"{self.output_filename}.csv", mode="data"))

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
  def batching(self, tokenized_dataset: Dataset) -> DataLoader:
    ...

  @abstractmethod
  def result_buffer(self) -> pd.DataFrame:
    ...

  @abstractmethod
  def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
    ...

  def execute(self, model, tokenizer: PreTrainedTokenizer, gen_params: dict[str, Any]) -> Any|list[Any]:
    tokenized_dataset = self.dataset.map(self.tokenization, new_fingerprint=self.benchname,
        fn_kwargs={"tokenizer": tokenizer, "prompt": self.define_prompt}, batched=True, 
        remove_columns=self.dataset.column_names, keep_in_memory=True, load_from_cache_file=False)
 
    tokenized_dataset.set_format(type="torch", columns=tokenized_dataset.column_names)

    logger.debug((
      f"Dataset tokenization: {repr(tokenized_dataset)}\n" 
      f"Cache files:\n"
      f"\toriginal: {self.dataset.cache_files} (should exist)\n"
      f"\ttokenized: {tokenized_dataset.cache_files} (should be empty)")
    )

    terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    batches = self.batching(tokenized_dataset)

    results_df = self.result_buffer()
    for i, batch in enumerate(batches):
      idx = batch[INDEX_COLUMN]
      input_ids = batch["input_ids"].to(device=model.device)
      attention_mask = batch["attention_mask"].to(device=model.device)
      batch_size = len(input_ids)
      
      logger.debug(f"Infering batch {i+1}/{len(batches)} of {batch_size} size...")
      outputs_ids = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, num_return_sequences=1, eos_token_id=terminators, **gen_params)

      for j, (idx_tensor, input_enc, output_enc) in enumerate(zip(idx, input_ids, outputs_ids)):
        idx = idx_tensor.item()
        input = tokenizer.decode(input_enc, skip_special_tokens=True)
        output = tokenizer.decode(output_enc[input_enc.shape[-1]:], skip_special_tokens=True)
        results_df.loc[i * batch_size + j, results_df.columns] = idx, input, output

    results_df.set_index(INDEX_COLUMN, inplace=True)

    if self._latent_mode == "output":
      self.save_latent_data(results_df)

    metrics_df = self.compute_metrics(results_df)
    metrics = metrics_df.agg(['mean', 'median']).T

    self.dataset.cleanup_cache_files()

    return metrics
  
