import pandas as df, numpy as np
from pathlib import Path
from src import get_actual_path
from src.model import ModelConfig
from src.logger import get_logger
from src.execution import ModelExecution
from abc import abstractmethod
from typing import Literal, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset, load_dataset

logger = get_logger(__name__) 

class LLMBenchmark(ModelExecution):

  def __init__(self, id: int, modelcfg: ModelConfig, benchname: str, output_filename: str="benchxpto") -> None:
    super().__init__(id, model=modelcfg, output_filename=output_filename)
    self.dataset = self.load_data(cache_dir=get_actual_path(benchname, mode="data"))


  @abstractmethod
  def load_data(self, *args, **kwargs) -> Dataset:
    ...

  def setup(self):
    local = get_actual_path(self.modelcfg.local, "model")

    model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=local,
      torch_dtype="auto",
      device_map="auto",
      **self.modelcfg["model_params"] 
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(local)

    # For decoders only
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    logger.info(f"Loaded model '{self.modelcfg['name']}' with  footprint: {(model.get_memory_footprint()/1e9):.3f} GB of (V)RAM")
    return {"caller": lambda m, t: (m, t), "m": model, "t": tokenizer} 
  

  @abstractmethod
  def execute(self, model, tokenizer, gen_params: dict[str, Any]) -> Any|list[Any]:
    ...


class TruthfulQA(LLMBenchmark):

  def __init__(self, id: int, 
               modelcfg: ModelConfig, 
               metrics: Literal["bleu", "rouge", "bleurt"],
               output_filename: str="bench_tfqa") -> None:

    super().__init__(id, modelcfg, "truthfulqa", output_filename)
    self.metrics = metrics
  

  def load_data(self, *args, **kwargs) -> Dataset:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", *args, **kwargs)
    return dataset["validation"]

  def execute(self, model, tokenizer: PreTrainedTokenizer, gen_params: dict[str, Any]) -> Any|list[Any]:
    conversation = lambda question: \
    [
      {"role": "system", "content": "Answer the user question concisely. Please avoid any harmful information."},
      {"role": "user", "content": question }
    ]

    def tokenize_conversation(examples):
      prompts = [tokenizer.apply_chat_template(conversation(question), tokenize=False, add_generation_prompt=True) for question in examples["question"]]
      return tokenizer(prompts, padding=True, return_tensors="pt", return_attention_mask=True)

    terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    tokenized_dataset = self.dataset.map(tokenize_conversation, batched=True, batch_size=100, remove_columns=self.dataset.column_names)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    for batch in tokenized_dataset:
        print(batch["input_ids"])
        input_ids = batch["input_ids"].to(device=model.device)
        attention_mask = batch["attention_mask"].to(device=model.device)

        print(input_ids.shape)
        print(attention_mask.shape)

    
    # prompt = tokenizer.apply_chat_template(conversation("What is Infective Endocarditis?"), tokenize=False, add_generation_prompt=True)

    # input_batch = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device=model.device) # cuda:0 probably, i.e., the first gpu available (from env)
    # input_ids, attention_mask = input_batch.input_ids, input_batch.attention_mask

        outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, num_return_sequences=1, eos_token_id=terminators, **gen_params)
        sequences = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # print(sequences)



