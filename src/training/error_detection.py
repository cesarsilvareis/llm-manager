import re, numpy as np
from src import get_actual_path, Path
from src.tasks import ED_MCQ
from src.model import ModelConfig
from src.tasks import Task, TaskType
from src.logger import get_logger
from src.loader import load_modelcfg_from_fs, load_model_from_hf, load_data_from_fs
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from operator import itemgetter
from transformers import (
  AutoTokenizer,
  PreTrainedTokenizer,
  DataCollatorWithPadding,
  TrainingArguments,
  Trainer,
  AdamW,
)

from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from peft import (
  LoraConfig,
  PeftModel,
  get_peft_model,
  prepare_model_for_kbit_training,
) 

from time import time
from src.training.balanced_trainer import BalancedTrainer
from src.training.data_preparation import EDStep1, EDStep4

logger = get_logger()


class Training_EDMCQ():

  # Training Data
  TRAIN_DATASET_HF_PATH = "openlifescienceai/medmcqa"
  TEST_DATASET_HF_PATH = "medalpaca/medical_meadow_medqa"

  TRAIN_SUBSET = "train"
  VALIDATION_SUBSET = "validation"
  TEST_SUBSET = "test"

  SOURCE_QUESTIONS = "question"
  SOURCE_OPTIONS = ["opa", "opb", "opc", "opd"]
  SOURCE_TRUE_OPTION = "cop"
  
  def __init__(self, modelcfg: ModelConfig, dataset: DatasetDict|None=None, to_save: str|Path|None=None) -> None:
    self._task = ED_MCQ()

    train_data = dataset or load_dataset(path=Training_EDMCQ.TRAIN_DATASET_HF_PATH) # train, val, (test)
    
    assert isinstance(train_data, DatasetDict) and all(s in train_data.column_names for s in (self.TRAIN_SUBSET, self.VALIDATION_SUBSET)) 

    if dataset is None:
      test_data = load_dataset(path=Training_EDMCQ.TEST_DATASET_HF_PATH)
      if isinstance(test_data, DatasetDict):
        test_data = test_data["train"]
      
      train_data = DatasetDict({
          self.TRAIN_SUBSET: train_data[self.TRAIN_SUBSET],
          self.VALIDATION_SUBSET: train_data[self.VALIDATION_SUBSET],
          self.TEST_SUBSET: test_data
        }
      )

    # p = EDStep1(self._task, train_data[self.VALIDATION_SUBSET], self.SOURCE_QUESTIONS, self.SOURCE_OPTIONS, self.SOURCE_TRUE_OPTION,
    #         initial_sizes={
    #           self.TRAIN_SUBSET: 37_500,
    #           self.VALIDATION_SUBSET: None,
    #           self.TEST_SUBSET: None
    #         },
    #   medical_subject_lbl=self._task.MEDICAL_SUBJECT_LABEL,
    #   split=self.VALIDATION_SUBSET
    # )

    # p = EDStep4(self._task, train_data, modelcfg=load_modelcfg_from_fs("llama2_7b"))
    # train_data = p.run()

    self._dataset = train_data

    if to_save:
      to_save = get_actual_path(to_save, mode="data")

      self._dataset.save_to_disk(to_save)
      self.train_dataset.to_csv(to_save.joinpath(f"train.csv"))
      self.validation_dataset.to_csv(to_save.joinpath(f"validation.csv"))
      self.test_dataset.to_csv(to_save.joinpath(f"test.csv"))

    print(self._dataset)

    # for split in self._dataset.column_names:
    #   self._dataset[split] = self._dataset[split].shuffle(self._task.SEED).select(range(10))
    self._dataset[self.TRAIN_SUBSET] = self._dataset[self.TRAIN_SUBSET].select(range(70_000))


    # Prepare pretrained model and its dependencies
    self._load_model(modelcfg)
    self._modelcfg = modelcfg
    self._already_prepared = False


  @property
  def train_dataset(this) -> Dataset:
    return this._dataset["train"]
  
  @property
  def validation_dataset(this) -> Dataset:
    return this._dataset["validation"]
  
  @property
  def test_dataset(this) -> Dataset:
    return this._dataset["test"]


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
  def tokenizing_data(dataset: DatasetDict, task: ED_MCQ, tokenizer: PreTrainedTokenizer, ctx_len) -> DatasetDict:
    
    column_names = dataset.column_names if isinstance(dataset, Dataset) else dataset["train"].column_names
    return dataset.map(function=task.tokenization, fn_kwargs={
        "tokenizer": tokenizer, 
        "question": task.QUESTION_LABEL,
        "answer": task.ANSWER_LABEL,
        "label": task.OUTPUT_LABEL,
        "ctx_len": ctx_len
      }, with_indices=False, batched=True, keep_in_memory=True,
        load_from_cache_file=False, remove_columns=column_names
    )

  def prepare_model_for_training(self, override: bool=True) -> PeftModel:
    self._pretrained_model.train()
    self._pretrained_model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(self._pretrained_model)
    config = LoraConfig(
      r=32,
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

    tokenized_dataset = self.tokenizing_data(self._dataset, self._task, self._tokenizer, ctx_len=2048)
    logger.info(f"Tokenization done! Dataset: {tokenized_dataset}")

    data_collator = DataCollatorWithPadding(self._tokenizer)

    training_args = TrainingArguments(
      output_dir=get_actual_path(finetuned_model_dir, mode="model"),
      logging_dir=get_actual_path(finetuned_model_dir, mode="log"),
      logging_steps=10,
      num_train_epochs=3,
      logging_strategy="epoch",
      eval_strategy="epoch",
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      bf16=True,
      max_grad_norm=1.0,
      save_total_limit=3,
      save_strategy="epoch",
      report_to=["tensorboard"],
      disable_tqdm=False,
      metric_for_best_model=self._task.get_max_metric().name,
      greater_is_better=True,
      data_seed=self._task.SEED,
    )

    to_train_model = self._pretrained_model
    if not self._already_prepared:
      to_train_model = self.prepare_model_for_training(override=True)
    
    optimizer = AdamW(to_train_model.parameters(), lr=3e-4, eps=1e-5, weight_decay=0.1, betas=[0.9, 0.95])
    lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, 
        num_training_steps=self.train_dataset.num_rows * training_args.num_train_epochs // training_args.per_device_train_batch_size, num_warmup_steps=2000,
        min_lr=1e-6
    )

    from torch import from_numpy
    train_class_weights = from_numpy(
      (1 - (tokenized_dataset["train"].to_pandas()["labels"].value_counts().sort_index()) / tokenized_dataset.num_rows["train"]).values
    ).float().to("cuda")

    eval_class_weights = from_numpy(
      (1 - (tokenized_dataset["validation"].to_pandas()["labels"].value_counts().sort_index()) / tokenized_dataset.num_rows["validation"]).values
    ).float().to("cuda")

    print(f"{train_class_weights=}, {eval_class_weights=}")

    trainer = Trainer(
      model=to_train_model,
      args=training_args,
      train_dataset=tokenized_dataset[self.TRAIN_SUBSET],
      eval_dataset=tokenized_dataset[self.VALIDATION_SUBSET],
      tokenizer=self._tokenizer,
      data_collator=data_collator,
      optimizers=(optimizer, lr_scheduler),
      compute_metrics=lambda x: self._task.compute_metrics_with_radar_for(x, 
                medical_subjects=self._dataset[self.VALIDATION_SUBSET]["subject_name"],
                save_path=get_actual_path(finetuned_model_dir, mode="log")
      )
    )

    # trainer = BalancedTrainer(
    #   model=to_train_model,
    #   args=training_args,
    #   train_dataset=tokenized_dataset["train"],
    #   eval_dataset=tokenized_dataset["validation"],
    #   tokenizer=self._tokenizer,
    #   data_collator=data_collator,
    #   optimizers=(optimizer, lr_scheduler),
    #   compute_metrics=self._task.compute_metrics,
    #   train_class_weights=train_class_weights,
    #   eval_class_weights=eval_class_weights
    # )
    
    logger.info(f"Training Optimizer:\n{trainer.optimizer}")
    logger.info(f"Training LR Scheduler:\n{trainer.lr_scheduler}")
    logger.info(f"Training SFT of model '{self._modelcfg['name']}'for ED of MCQs...")
    logger.info(f"Dataset shapes: train={self.train_dataset.num_rows}, validation={self.validation_dataset.num_rows}, test={self.test_dataset.num_rows}")

    to_train_model.print_trainable_parameters()
    
    if input("Initiate Training? (Y|N) ").strip().lower() not in ["y", "Y"]:
      return
    
    stime = time()
    trainer.train()
    dtime = time() - stime
    logger.info(f"SFT for ED of MCQs have finished in {dtime/3600:.2f}h")

    # url = trainer.push_to_hub(f'SFT EDCMQC have finished for {self._modelcfg["name"]}')
    # logger.info(f"URL for accessing the model online (in HF):\n{url}")

  