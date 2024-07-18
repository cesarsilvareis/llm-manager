import numpy as np
from src import get_actual_path, Path
from src.tasks import ED_MCQ
from src.model import ModelConfig
from src.tasks import Task, TaskType
from src.logger import get_logger
from src.loader import load_model_from_hf
from datasets import Dataset, DatasetDict, load_dataset
from operator import itemgetter
from transformers import (
  AutoTokenizer,
  PreTrainedTokenizer,
  DataCollatorWithPadding,
  TrainingArguments,
  Trainer
)
from peft import (
  LoraConfig,
  PeftModel,
  get_peft_model,
  prepare_model_for_kbit_training,
) 

from time import time

logger = get_logger()



class Training_EDMCQ():

  # DATA SOURCE
  DATASET_HF_PATH = "openlifescienceai/medmcqa"

  FT_QUESTIONS = "question"
  FT_OPTIONS = ["opa", "opb", "opc", "opd"]
  FT_GROUND_TRUTH_OPTION = "cop"
  FT_ANSWERS = "answer"
  FT_LABEL = "label"

  DATASET_SIZES = {
    "train": 15_000,
    "validation": 2_500,
    "test": 5_000
  }
  
  def __init__(self, modelcfg: ModelConfig, dataset: Dataset|DatasetDict|None=None, to_save: str|Path|None=None) -> None:
    self._task = ED_MCQ()

    self._dataset = dataset or load_dataset(path=Training_EDMCQ.DATASET_HF_PATH).select_columns(
      [
        Training_EDMCQ.FT_QUESTIONS, 
        *Training_EDMCQ.FT_OPTIONS, 
        Training_EDMCQ.FT_GROUND_TRUTH_OPTION 
      ]
    )

    # Selecting portion of the dataset randomly
    if isinstance(self._dataset, DatasetDict):
      for data_ideo, size in Training_EDMCQ.DATASET_SIZES.items():
          self._dataset[data_ideo] = \
            self._dataset[data_ideo].shuffle(seed=self._task.SEED) \
                                            .select(range(size))
    else:
      self._dataset = self._dataset.shuffle(seed=self._task.SEED).select(range(size))
    
    # Data science...
    self._preprocessing_data(to_save)

    # Prepare pretrained model and its dependencies
    self._load_model(modelcfg)
    self._modelcfg = modelcfg
    self._already_prepared = False


  @property
  def train_dataset(this) -> Dataset:
    return this._dataset if isinstance(this._dataset, Dataset) else this._dataset["train"]
  
  def _preprocessing_data(self, to_save: str|Path|None=None):

    def split_answers(examples):
      separator = zip(
        examples[Training_EDMCQ.FT_QUESTIONS], 
        *itemgetter(*Training_EDMCQ.FT_OPTIONS)(examples), 
        examples[Training_EDMCQ.FT_GROUND_TRUTH_OPTION]
      )

      qa_groups = []
      for question, *options, label in separator:
          qa_groups.extend((question, opt, self._task.get_label(i, label, return_id=True)[0])
                            for i, opt in enumerate(options))
        
      questions, answers, labels = zip(*qa_groups)

      return {
          Training_EDMCQ.FT_QUESTIONS: list(questions), 
          Training_EDMCQ.FT_ANSWERS: list(answers), 
          Training_EDMCQ.FT_LABEL: list(labels)
      }

    self._dataset = self._dataset.map(split_answers, with_indices=False, batched=True, batch_size=8, 
                      load_from_cache_file=False, keep_in_memory=True, remove_columns=self.train_dataset.column_names)

    if to_save:
      to_save = get_actual_path(to_save, mode="data")

      self._dataset.save_to_disk(to_save)
      self.train_dataset.to_csv(to_save.joinpath("train.csv"))
      if isinstance(self._dataset, DatasetDict):
        self._dataset["test"].save_to_disk(to_save.joinpath("test"))
        self._dataset["test"].to_csv(to_save.joinpath("test.csv"))

  def _load_model(self, modelcfg: ModelConfig):
    load_model_from_hf(modelcfg)

    checkpoint_local = get_actual_path(modelcfg.local, mode="model")
    from torch import bfloat16
    self._pretrained_model = self._task.get_pretrained_model_forme(
        checkpoint_local, torch_dtype = bfloat16, num_labels = self._task.num_labels
    )

    logger.info(f"Model has been loaded! {self._pretrained_model}")

    self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_local, add_prefix_space=True)

    # Configurations for our decoder-only transformer
    self._tokenizer.padding_side = "left"
    self._tokenizer.pad_token = self._tokenizer.eos_token
    self._pretrained_model.resize_token_embeddings(len(self._tokenizer))
    self._pretrained_model.config.pad_token_id = self._pretrained_model.config.eos_token_id

  @staticmethod
  def tokenizing_data(dataset: DatasetDict, task: Task, tokenizer: PreTrainedTokenizer, ctx_len: int=2000) -> DatasetDict:
    
    column_names = dataset.column_names if isinstance(dataset, Dataset) else dataset["train"].column_names
    return dataset.map(function=task.tokenization, fn_kwargs={
        "tokenizer": tokenizer, 
        "question": Training_EDMCQ.FT_QUESTIONS,
        "answer": Training_EDMCQ.FT_ANSWERS,
        "label": Training_EDMCQ.FT_LABEL,
        "ctx_len": ctx_len
      }, with_indices=False, batched=True, keep_in_memory=True, load_from_cache_file=False,
        remove_columns=column_names
    )

  def prepare_model_for_training(self, override: bool=True) -> PeftModel:
    self._pretrained_model.train()
    self._pretrained_model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(self._pretrained_model)
    config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["q_proj"], # encountered in llama 2 base specification
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, config)
    if override:
      self._pretrained_model = model
      import torch
      torch.cuda.ipc_collect()
      torch.cuda.empty_cache()
      self._already_prepared = True

    return model


  def run_sft(self, finetuned_model_dir: str|Path):

    tokenized_dataset = self.tokenizing_data(self._dataset, self._task, self._tokenizer, ctx_len=2000)
    logger.info(f"Tokenization done! Dataset: {tokenized_dataset}")

    data_collator = DataCollatorWithPadding(self._tokenizer)

    result_model_name = f"sft-mcqed-{self._modelcfg.filename}"
    training_args = TrainingArguments(
      output_dir=get_actual_path(result_model_name, mode="model"),
      logging_dir=get_actual_path(result_model_name, mode="log"),
      logging_steps=10,
      num_train_epochs=10,
      logging_strategy="epoch",
      eval_strategy="epoch",
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      bf16=True,
      max_grad_norm=1,
      learning_rate=3e-4,
      lr_scheduler_type="cosine", 
      weight_decay=0.1,
      save_total_limit=2,
      save_strategy="epoch",
      report_to=["tensorboard"],
      load_best_model_at_end=True,
      metric_for_best_model=self._task.max_metric.name,
      greater_is_better=True,
      gradient_accumulation_steps=4,
      data_seed=self._task.SEED,
      optim="paged_adamw_8bit"
    )
    logger.info(f"Training Arguments:\n{training_args}")

    self._pretrained_model.config.use_cache = False
    trainer = Trainer(
      model=self._pretrained_model,
      args=training_args,
      train_dataset=tokenized_dataset["train"],
      eval_dataset=tokenized_dataset["validation"],
      tokenizer=self._tokenizer,
      data_collator=data_collator,
      compute_metrics=self._task.compute_metrics
    )

    # model_size = sum(p.numel() for p in pretrained_model.parameters())
    # print(f"Training model with {model_size/1e9:.2f}B parameters")
    
    to_train_model = self._pretrained_model
    if not self._already_prepared:
      to_train_model = self.prepare_model_for_training(override=True)
    
    to_train_model.print_trainable_parameters()

    if input("Initiate Training? (Y|N)").lower() not in ["y", "Y"]:
      return
    
    logger.info(f"Training SFT of model '{self._modelcfg['name']}'for ED of MCQs...")
    stime = time()
    trainer.train()
    dtime = time() - stime
    logger.info(f"SFT for ED of MCQs have finished in {dtime/3600:.2f}h")

    to_train_model.config.use_cache = True

    testset = tokenized_dataset["test"]

    test_predictions = trainer.predict(testset)

    logger.info(
      f"Testing:\n"
        f"\tpredition: {test_predictions.metrics};\n"
        f"\tevaluation: {trainer.evaluate(testset)}"
    )
    logger.debug(f"Predictions:\n{np.stack((np.argmax(test_predictions.predictions, axis=-1), test_predictions.label_ids), axis=1)}")

    finetuned_model_dir = get_actual_path(finetuned_model_dir, mode="model")

    logger.info(f"Saving the obtained model in {finetuned_model_dir}")
    trainer.save_model(output_dir=finetuned_model_dir)
    url = trainer.push_to_hub("SFT EDCMQC have finished for meditron7b")
    logger.info(f"URL for accessing the model online (in HF):\n{url}")

  