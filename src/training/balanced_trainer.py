from typing import Callable, Dict, List, Tuple, Any
from datasets.arrow_dataset import Dataset
from torch._tensor import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers import Trainer
from torch import nn
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

class BalancedTrainer(Trainer):

  def __init__(self, class_weights, model: PreTrainedModel | nn.Module = None, args: TrainingArguments = None, data_collator: Any | None = None, train_dataset: Dataset | IterableDataset | Dataset | None = None, eval_dataset: Dataset | Dict[str, Dataset] | Dataset | None = None, tokenizer: PreTrainedTokenizerBase | None = None, model_init: Callable[[], PreTrainedModel] | None = None, compute_metrics: Callable[[EvalPrediction], Dict] | None = None, callbacks: List[TrainerCallback] | None = None, optimizers: Tuple[Optimizer, LambdaLR] = ..., preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None, ):
    super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
    self._class_weights = class_weights

  def compute_loss(self, model, inputs, return_outputs=False):
    # Feed inputs to the model and extract logits
    outputs = model(**inputs)
    logits = outputs.get("logits")

    # Extract labels
    labels = inputs.get("labels")
    
    # Define loss function with class weights 
    loss_func = nn.CrossEntropyLoss(weight=self._class_weights)

    # Compute loss
    loss = loss_func(logits, labels)
    return (loss, outputs) if return_outputs else loss