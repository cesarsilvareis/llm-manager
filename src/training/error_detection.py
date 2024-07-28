import re, numpy as np
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

logger = get_logger()



class Training_EDMCQ():

  # Training Data
  TRAIN_DATASET_HF_PATH = "openlifescienceai/medmcqa"
  TRAIN_SUBSET = "train"
  VALIDATION_SUBSET = "validation"
  TRAIN_QUESTIONS = "question"
  TRAIN_OPTIONS = ["opa", "opb", "opc", "opd"]
  TRAIN_GROUND_TRUTH_OPTION = "cop"
  

  # Testing Data
  TEST_DATASET_HF_PATH = "medalpaca/medical_meadow_medqa"
  TEST_SUBSET = "test"
  TEST_MCQ = "input" # question + options; i.e. MCQ
  TEST_GROUND_TRUTH_OPTION = "output"

  
  # Common Features
  CF_LABEL = "label"
  CF_ANSWERS = "answer"
  

  DATASET_SIZES = {
    "train":       69_858,
    "validation":  5_000,
    "test":        15_000
  }
  
  def __init__(self, modelcfg: ModelConfig, train_data: Dataset|DatasetDict|None=None, test_data: Dataset|DatasetDict|None=None, do_preprocessing: bool=False, to_save: str|Path|None=None) -> None:
    self._task = ED_MCQ()

    # Parsing training data to a separated dataset
    _train_dataset = train_data or load_dataset(path=Training_EDMCQ.TRAIN_DATASET_HF_PATH).select_columns(
      [
        Training_EDMCQ.TRAIN_QUESTIONS, 
        *Training_EDMCQ.TRAIN_OPTIONS, 
        Training_EDMCQ.TRAIN_GROUND_TRUTH_OPTION 
      ]
    )

    if isinstance(_train_dataset, DatasetDict):
      _eval_dataset = _train_dataset[Training_EDMCQ.VALIDATION_SUBSET]
      _train_dataset = _train_dataset[Training_EDMCQ.TRAIN_SUBSET]
    else: # assuming just the training dataset is given
      split = Training_EDMCQ.DATASET_SIZES["validation"]
      _train_dataset, _eval_dataset = _train_dataset[:-split], _train_dataset[-split:]

    # Parsing testing data to a separated dataset
    _test_dataset = test_data or load_dataset(path=Training_EDMCQ.TEST_DATASET_HF_PATH).select_columns(
      [
        Training_EDMCQ.TEST_MCQ, 
        Training_EDMCQ.TEST_GROUND_TRUTH_OPTION 
      ]
    )

    if isinstance(_test_dataset, DatasetDict):
      _test_dataset = _test_dataset[Training_EDMCQ.TEST_SUBSET]
  
    self._dataset = DatasetDict({
       "train": _train_dataset,
       "validation": _eval_dataset,
       "test": _test_dataset 

    })
    print(self._dataset)


    # Data science...
    if do_preprocessing:
      self._preprocessing_data(to_save)
    else:
      for split in self._dataset.column_names:
        self._dataset[split] = self._dataset[split].shuffle(seed=self._task.SEED) \
                                                   .select(range(Training_EDMCQ.DATASET_SIZES[split]))

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
  

  def _preprocessing_data(self, to_save: str|Path|None=None):
    from bleurt import score
    scorer = score.BleurtScorer()

    def is_unique_inside(new_question: str, qa_groups: list[str], threshold=0.5):
      for question, *_ in qa_groups:
        if scorer.score(references=[question], candidates=[new_question])[0] > threshold:
          return False
      
      return True


    # Clean training & validation data
    def associate_answers(examples):
      separator = zip(
        examples[Training_EDMCQ.TRAIN_QUESTIONS], 
        *itemgetter(*Training_EDMCQ.TRAIN_OPTIONS)(examples), 
        examples[Training_EDMCQ.TRAIN_GROUND_TRUTH_OPTION]
      )

      qa_groups = []
      for question, *options, label in separator:
        if not is_unique_inside(question, qa_groups):
          continue

        labels_ids = [(opt, self._task.get_label(i, label, return_id=True)[0]) for i, opt in enumerate(options)]
        true_option = next(((opt, label_id) for opt, label_id in labels_ids if label_id == self._task.labels2ids[self._task.CORRECT_LABEL]), None)
        false_option = next(((opt, label_id) for opt, label_id in labels_ids if label_id == self._task.labels2ids[self._task.INCORRECT_LABEL]), None)
        if None in [true_option, false_option]:
          continue

        qa_groups.extend((question, opt, label_id) for opt, label_id in [true_option, false_option])
        # qa_groups.extend((question, opt, self._task.get_label(i, label, return_id=True)[0]) 
        #                  for i, opt in enumerate(options))
        
      questions, answers, labels = zip(*qa_groups)

      return {
          Training_EDMCQ.TRAIN_QUESTIONS: list(questions), 
          Training_EDMCQ.CF_ANSWERS: list(answers), 
          Training_EDMCQ.CF_LABEL: list(labels)
      }

    for split in ["train", "validation"]:
      self._dataset[split] = self._dataset[split]\
                                       .map(associate_answers, 
            with_indices=False, batched=True, batch_size=8, load_from_cache_file=False, 
            keep_in_memory=True, remove_columns=self._dataset[split].column_names)\
                                       .shuffle(seed=self._task.SEED) \
                                       .select(range(Training_EDMCQ.DATASET_SIZES[split])) \

      
    # Clean testing
    def split_answers(examples):
      mcq_pattern = r"Q:(.*)\{'A':\s*['\"](.*)['\"], 'B':\s*['\"](.*)['\"], 'C':\s*['\"](.*)['\"], 'D':\s*['\"](.*)['\"], 'E':\s*['\"](.*)['\"]\}"
      cop_pattern = r"[A-Z]: (.*)"
      mcq_extractor, cop_extractor = re.compile(mcq_pattern, re.DOTALL), re.compile(cop_pattern, re.DOTALL)

      separator = zip(examples[Training_EDMCQ.TEST_MCQ], examples[Training_EDMCQ.TEST_GROUND_TRUTH_OPTION])
      qa_groups = []
      for mcq, cop_id in separator:
        question, *options = mcq_extractor.search(mcq).groups()
        cop = cop_extractor.search(cop_id).group(1).strip()
        labels_ids = [(opt, self._task.get_label(opt, cop)) for opt in options]
        true_option = next(((opt, label) for opt, label in labels_ids if label == self._task.CORRECT_LABEL), None)
        false_option = next(((opt, label) for opt, label in labels_ids if label == self._task.INCORRECT_LABEL), None)
        if None in [true_option, false_option]:
          continue

        qa_groups.extend((question.strip(), opt, self._task.labels2ids[label]) for opt, label in [true_option, false_option])

      questions, answers, labels = zip(*qa_groups)

      return {
          Training_EDMCQ.TRAIN_QUESTIONS: list(questions), 
          Training_EDMCQ.CF_ANSWERS: list(answers), 
          Training_EDMCQ.CF_LABEL: list(labels)
      }
    
    self._dataset["test"] = self.test_dataset.map(split_answers,
            with_indices=False, batched=True, batch_size=8, load_from_cache_file=False, 
            keep_in_memory=True, remove_columns=self.test_dataset.column_names) \
                                             .shuffle(seed=self._task.SEED) \
                                             .select(range(Training_EDMCQ.DATASET_SIZES["test"])) \

    if to_save:
      to_save = get_actual_path(to_save, mode="data")

      self._dataset.save_to_disk(to_save)
      self.train_dataset.to_csv(to_save.joinpath(f"train.csv"))
      self.validation_dataset.to_csv(to_save.joinpath(f"validation.csv"))
      self.test_dataset.to_csv(to_save.joinpath(f"test.csv"))


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
        "question": Training_EDMCQ.TRAIN_QUESTIONS,
        "answer": Training_EDMCQ.CF_ANSWERS,
        "label": Training_EDMCQ.CF_LABEL,
        "ctx_len": ctx_len
      }, with_indices=False, batched=True, keep_in_memory=True, load_from_cache_file=False,
        remove_columns=column_names
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

    tokenized_dataset = self.tokenizing_data(self._dataset, self._task, self._tokenizer, ctx_len=2000)
    logger.info(f"Tokenization done! Dataset: {tokenized_dataset}")

    data_collator = DataCollatorWithPadding(self._tokenizer)

    training_args = TrainingArguments(
      output_dir=get_actual_path(finetuned_model_dir, mode="model"),
      logging_dir=get_actual_path(finetuned_model_dir, mode="log"),
      logging_steps=10,
      num_train_epochs=5,
      logging_strategy="epoch",
      eval_strategy="epoch",
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      bf16=True,
      max_grad_norm=1.0,
      # save_total_limit=2,
      warmup_ratio=0.1,
      save_strategy="epoch",
      report_to=["tensorboard"],
      load_best_model_at_end=True,
      metric_for_best_model=self._task.get_max_metric().name,
      greater_is_better=True,
      data_seed=self._task.SEED,
    )

    logger.info(f"Training Arguments:\n{training_args}")

    to_train_model = self._pretrained_model
    if not self._already_prepared:
      to_train_model = self.prepare_model_for_training(override=True)
    
    optimizer = AdamW(to_train_model.parameters(), lr=2e-5, eps=1e-5, weight_decay=0.1, betas=[0.9, 0.95])
    lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, 
        num_training_steps=self.train_dataset.num_rows * training_args.num_train_epochs // training_args.per_device_train_batch_size, num_warmup_steps=2000,
        min_lr=2e-6
    )

    from torch import from_numpy
    class_weights = from_numpy(
      (1 - (tokenized_dataset["train"].to_pandas()["labels"].value_counts().sort_index()) / tokenized_dataset.num_rows["train"]).values
    ).float().to("cuda")

    print(f"{class_weights=}")

    trainer = Trainer(
      model=to_train_model,
      args=training_args,
      train_dataset=tokenized_dataset["train"],
      eval_dataset=tokenized_dataset["validation"],
      tokenizer=self._tokenizer,
      data_collator=data_collator,
      optimizers=(optimizer, lr_scheduler),
      compute_metrics=self._task.compute_metrics
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
    #   class_weights=class_weights
    # )

    # model_size = sum(p.numel() for p in pretrained_model.parameters())
    # print(f"Training model with {model_size/1e9:.2f}B parameters")
    
    self._pretrained_model.config.use_cache = False
    
    logger.info(f"Training SFT of model '{self._modelcfg['name']}'for ED of MCQs...")
    logger.info(f"Dataset shapes: train={self.train_dataset.num_rows}, validation={self.validation_dataset.num_rows}, test={self.test_dataset.num_rows}")

    to_train_model.print_trainable_parameters()
    
    if input("Initiate Training? (Y|N)").lower() not in ["y", "Y"]:
      return
    
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

    url = trainer.push_to_hub("SFT EDCMQC have finished for meditron7b")
    logger.info(f"URL for accessing the model online (in HF):\n{url}")

  