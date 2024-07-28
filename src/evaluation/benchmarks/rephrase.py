import evaluate, pandas as pd
from typing import Any, Literal
from src import get_actual_path
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
from torch.utils.data import DataLoader

from src.model import ModelConfig
from src.evaluation import LLMBenchmark, INDEX_COLUMN

class QuestionRephrase(LLMBenchmark):

  EXAMPLES = [
("""
Q: "Abnormal vascular patterns seen with colposcopy in case of cervical intraepithelial neoplasia are all except?"
""", # Example 1
"""
R: "What abnormal vascular pattern is not seen with colposcopy in case of cervical intraepithelial neoplasia?",
"""),
("""
Q: "All of the following are surgical options for morbid obesity except -"
""", # Example 2
"""
R: "Give me a surgical option that is not for morbid obesity."
"""),
("""
Q: "Following endaerectomy on the right common carotid, a patient is found to be blind in the right eye. It is appears that a small thrombus embolized during surgery and lodged in the aery supplying the optic nerve. Which aery would be blocked?"
""", # Example 3
"""
R: "Following endaerectomy on the right common carotid, a patient is found to be blind in the right eye. It is appears that a small thrombus embolized during surgery and lodged in the aery supplying the optic nerve. Which aery would be blocked?"
"""),
("""
Q: "Which one of the following is an environmental factor associated with mental illness?a)  Emotional stressb)  Frustrationc)  Broken homed)  Anxiety"
""", # Example 4
"""
R: "Which one of the following is an environmental factor associated with mental illness?a)  Emotional stress; b)  Frustration; c)  Broken home; or d)  Anxiety?"
"""),
("""
Q: "Of the various modalities used in the treatment of re-threatening effects of hyperkalemia which one of the following as the most rapid onset of action?"
""", # Example 5
""" 
R: "Of the various modalities used in the treatment of re-threatening effects of hyperkalemia, give me one with rapid onset of action."
"""),
("""
Q: "Cattle truck appearance on fundus examination is a feature of: (Repeat)"
""", # Example 6
""" 
R: "Cattle truck appearance on fundus examination is a feature of _______."
"""),
("""
Q: "pseudorosettes are seen in -"
""", # Example 6
""" 
R: "Pseudorosettes are seen in ________."
""")
]

  SYSTEM_PROMPT = """
You are given with a question potentially not connected grammarly to short answers.
Generate the rephrased version of the user's question enclosed between double quotes. 
Avoid being verbose, nor change the meaning of the problem. Do NOT answer to your question.

"""

  def __init__(self, id: int, 
               modelcfg: ModelConfig,
               outputfile: str="",
               save_latents: Literal["output"]="output") -> None:

    super().__init__(id, modelcfg, outputfile, save_latents)

  def load_data(self, *args, **kwargs) -> Dataset:
    dataset = load_from_disk(get_actual_path("preproc_mcqed_data_v3", mode="data"))
    return dataset["train"].select(range(7000))
  
  def select_data(self):
    return self.dataset.shuffle(seed=7 * self.REPEAT).select(range(25))

  def define_prompt(self, question: str, answer: str):
    return [
      {"role": "system", "content": QuestionRephrase.SYSTEM_PROMPT},
      *[item for sublist in [({"role": "user", "content": user}, {"role": "assistant", "content": assistant}) for user, assistant in QuestionRephrase.EXAMPLES] for item in sublist],
      {"role": "user", "content": f'Q: "{question}"\nR: '}
    ]

  def tokenize(self, tokenizer: PreTrainedTokenizer, prompt, examples) -> BatchEncoding:
    prompts = [tokenizer.apply_chat_template(prompt(question, answer), tokenize=False, add_generation_prompt=True) for question, answer in zip(examples["question"], examples["answer"])]
    return tokenizer(prompts, padding=True, return_tensors="pt", return_attention_mask=True)
  
  def batching(self, tokenized_dataset: Dataset, tokenizer) -> DataLoader:
    return DataLoader(tokenized_dataset, shuffle=True, batch_size=15, collate_fn=DataCollatorWithPadding(tokenizer))

  def result_buffer(self) -> pd.DataFrame:
    return pd.DataFrame(columns=[INDEX_COLUMN, "prompt", "reframed_question"])
  
  def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
    return None
  
  def get_score(self, results) -> float:
    return None