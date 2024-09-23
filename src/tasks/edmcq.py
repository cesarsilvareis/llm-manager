import numpy as np
from collections import ChainMap
from evaluate import EvaluationModule, load


from src.tasks import Task, TaskType, Dataset, DatasetDict, PreTrainedTokenizer
from transformers import EvalPrediction

class ED_MCQ(Task):

  SEED = 17

  CORRECT_LABEL = "Correct"
  INCORRECT_LABEL = "Incorrect"

  METRICS = [
    load("f1"),  # the upper the chosen for scoring checkpoints
    load("accuracy"),
    load("recall"),
    load("precision"),
  ]

  QUESTION_LABEL = "question"
  ANSWER_LABEL = "answer"
  MEDICAL_SUBJECT_LABEL = "subject_name"
  OUTPUT_LABEL = "label"



  def __init__(self) -> None:
    from src.training import Training_EDMCQ
    super().__init__(TaskType.SEQ_CLS, trainer_scheme=Training_EDMCQ)

  @property
  def labels2ids(this):
    return {
      ED_MCQ.CORRECT_LABEL:   0,  # default case (False)
      ED_MCQ.INCORRECT_LABEL: 1   # alarming case (True)
    }
  
  @property
  def ids2labels(this):
    return { i : l for l, i in this.labels2ids.items() }
  
  @property
  def num_labels(this):
    return len(this.labels2ids)
  
  def get_max_metric(self) -> EvaluationModule:
    return self.METRICS[0]

  def get_label(self, opi: int|str, opc: int|str, return_id: bool=False) -> str:
    label = ED_MCQ.CORRECT_LABEL if opi == opc else ED_MCQ.INCORRECT_LABEL
    return (self.labels2ids[label], label) if return_id else label


  def get_parameters(self) -> dict:
    return { "label2id" : self.labels2ids, "id2label": self.ids2labels }

  def tokenization(self, examples, tokenizer: PreTrainedTokenizer, 
      question: str, answer: str, label: str, ctx_len: int=252) -> Dataset | DatasetDict:
    
    return { "labels": examples[label], **tokenizer(
      examples[question], examples[answer], # QA pairwise input
      truncation=True, max_length=ctx_len, return_tensors="np", return_overflowing_tokens=False)
    }
  

  def compute_metrics(self, eval_preds: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1) 
    return dict(ChainMap(*(metric.compute(predictions=predictions, references=labels) 
                           for metric in ED_MCQ.METRICS)))
  

  def compute_metrics_with_radar_for(self, eval_preds: EvalPrediction, medical_subjects, save_path):

    def draw_radar_diagram():
      import matplotlib.pyplot as plt
      
      labels = list(subject_wise_metrics.keys())
      metrics: dict[str, list] = { m.name: [] for m in ED_MCQ.METRICS }

      # Extract values for each subject
      for subject in labels:
          for metric, res  in subject_wise_metrics[subject].items():
            metrics[metric].append(res)
      
      metrics = {m : r + r[:1] for m, r in metrics.items()}
      labels += labels[:1]  # Loop around to close the radar chart

      # Create the radar chart
      angles = np.linspace(0, 2 * np.pi, len(labels) - 1, endpoint=False).tolist()
      angles += angles[:1]

      _, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

      colours = ['orange', 'green', 'blue', 'red'][:len(metrics)]

      # Plot areas
      for i, (metric, res) in enumerate(metrics.items()): 
        ax.fill(angles, res, color=colours[i], alpha=0.15, label=metric.capitalize())

      for i, (metric, res) in enumerate(metrics.items()):   
        ax.plot(angles, res, color=colours[i], linewidth=2, linestyle="solid")

      # Set labels and title
      ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
      ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
      ax.set_xticks(angles[:-1])
      ax.set_xticklabels(labels[:-1])

      plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

      # Show plot
      plt.tight_layout()
      
      # Save the figure
      plt.savefig(f"{save_path}_{self._epoch}.png", format='png', bbox_inches='tight')

      # Clear the plot after saving
      plt.clf()


    subject_wise_metrics = { subject: 
      self.compute_metrics(
        tuple(e[np.array(medical_subjects) == subject] for e in eval_preds)
        ) for subject in ["Medicine", "Anatomy", "Pathology", "Microbiology", "Radiology", "Surgery", "Pharmacology"]
      }

    subject_wise_metrics["ALL"] = self.compute_metrics(eval_preds)

    draw_radar_diagram()

    self._epoch += 1

    return subject_wise_metrics["ALL"] # to overall evaluation
