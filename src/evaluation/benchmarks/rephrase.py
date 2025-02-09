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
""", # Example 1: multi-answered exclusion
"""
F: "What abnormal vascular pattern is not seen with colposcopy in case of cervical intraepithelial neoplasia?",
"""),
("""
Q: "All of the following are surgical options for morbid obesity except -"
""", # Example 2: direct + multi-answered exclusion
"""
F: "Provide a surgical option that is not for morbid obesity."
"""),
("""
Q: "Following endaerectomy on the right common carotid, a patient is found to be blind in the right eye. It is appears that a small thrombus embolized during surgery and lodged in the aery supplying the optic nerve. Which aery would be blocked?"
""", # Example 3: subject focus
"""
F: "Following endaerectomy on the right common carotid, a patient is found to be blind in the right eye. It is appears that a small thrombus embolized during surgery and lodged in the aery supplying the optic nerve. Which aery would be blocked?"
"""),
("""
Q: "Of the various modalities used in the treatment of re-threatening effects of hyperkalemia, which one of the following as the most rapid onset of action?"
""", # Example 5: multi-answered
""" 
F: "Of the various modalities used in the treatment of re-threatening effects of hyperkalemia, give one with rapid onset of action."
"""),
("""
Q: "Cattle truck appearance on fundus examination is a feature of: (Repeat)"
""", # Example 6: remove unnecessary stuff 
""" 
F: "Cattle truck appearance on fundus examination is a feature of _______."
"""),
("""
Q: "pseudorosettes are seen in -"
""", # Example 7: direct
""" 
F: "Where pseudorosettes are seen?"
"""),
("""
Q: "A 48-year-old man presents with 3 weeks of fever, fatigue, and shortness of breath. He has a history of ""nasal allergies"" and asthma, which have been poorly controlled in the past month. Two days prior to presentation, he developed weakness in his left foot and it now ""drags"" when he walks. On examination, his blood pressure is 165/90 mm Hg, pulse 100/min, respirations 20/min, and lungs have bilateral expiratory wheezes. There is left foot drop, and the rest of the neurologic examination is normal. Laboratory evaluation reveals ESN of 90 mm/h, WBC of 14,000/mL with 10% eosinophils, and 1+ proteinuria. A CXN shows bilateral pulmonary infiltrates.For the above patient with vasculitis syndrome, select the most likely diagnosis."
""", # Example 8: context
""" 
F: "A 48-year-old man presents with 3 weeks of fever, fatigue, and shortness of breath. He has a history of ""nasal allergies"" and asthma, which have been poorly controlled in the past month. Two days prior to presentation, he developed weakness in his left foot and it now ""drags"" when he walks. On examination, his blood pressure is 165/90 mm Hg, pulse 100/min, respirations 20/min, and lungs have bilateral expiratory wheezes. There is left foot drop, and the rest of the neurologic examination is normal. Laboratory evaluation reveals ESN of 90 mm/h, WBC of 14,000/mL with 10% eosinophils, and 1+ proteinuria. A CXN shows bilateral pulmonary infiltrates. What is the most likely diagnosis for this patient with vasculitis syndrome?"
"""),
("""
Q: "A body is brought to you for autopsy. The suspected cause of death is drowning. Which of the following would you NOT expect to find in this body?"
""", # Example 9: context + multi-answered
""" 
F: "A body is brought to you for autopsy. The suspected cause of death is drowning. What are you not expecting to find in this body?"
"""),
("""
Q: "One of the following is not an amino acid:"
""", # Example 9: context + multi-answered
""" 
F: "Provide a molecule that is not an amino acid."
"""),
("""
Q: "Which of the following organism is seen in a patient of left-sided endocarditis involving the mitral valve?"
""", # Example 9: context + multi-answered
""" 
F: "What is the organism seen in a patient of left-sided endocarditis involving the mitral valve?"
"""),
("""
Q: "A 26-year-old primigravida diagnosed with severe rheumatic heart disease (Mitral stenosis with mitral regurgitation) is in early labour. The obstetrician wants to try a normal labour. Which of the following is the best labour analgesia for this patient?"
""", # Example 9: context + multi-answered
""" 
F: "A 26-year-old primigravida diagnosed with severe rheumatic heart disease (Mitral stenosis with mitral regurgitation) is in early labour. The obstetrician wants to try a normal labour. What is the best labour analgesia for this patient?"
"""),
("""
Q: "Which of the following occurs along with glucose transpo into a cell:"
""", # Example 9: context + multi-answered
""" 
F: "What can you say that occurs along with glucose transport into a cell?"
"""),
("""
Q: "In which of the following heart diseases maternal mortality is found to be highest ?"
""", # Example 9: context + multi-answered
""" 
F: "Give a heart disease with higher maternal mortality."
"""),
("""
Q: "A 15-month-old boy is brought the pediatrician for immunizations and assessment. His parents report that he is eating well and produces several wet diapers every day. He is occasionally fussy, but overall a happy and curious child. The boy was born at 39 weeks gestation via spontaneous vaginal delivery On physical examination his vital signs are stable. His weight and height are above the 85th percentile for his age and sex. On chest auscultation, the pediatrician detects a loud harsh holosystolic murmur over the left lower sternal border. The first and second heart sounds are normal. An echocardiogram confirms the diagnosis of the muscular ventricular septal defect without pulmonary hypertension. Which of the following is the best management strategy for this patient?"
""", # Example 9: context + multi-answered
""" 
F: "A 15-month-old boy is brought the pediatrician for immunizations and assessment. His parents report that he is eating well and produces several wet diapers every day. He is occasionally fussy, but overall a happy and curious child. The boy was born at 39 weeks gestation via spontaneous vaginal delivery On physical examination his vital signs are stable. His weight and height are above the 85th percentile for his age and sex. On chest auscultation, the pediatrician detects a loud harsh holosystolic murmur over the left lower sternal border. The first and second heart sounds are normal. An echocardiogram confirms the diagnosis of the muscular ventricular septal defect without pulmonary hypertension. What is the best management strategy for this patient?"
"""),
]

  SYSTEM_PROMPT = """
You are given with the user's multiple-choice question (MCQ) potentially not \
logically connected to single answers when the universe of options is omitted. \
Your task is to remove its MCQ style (i.e., any "which of the following"-like \
references) to create a single-answer question. For that: provide all given \
necessary context (e.g., a clinical case); avoid being verbose; keep the meaning \
of the problem; and do NOT respond to the question. Just provide the fixed version \
after the "F" mark. 
"""

  def __init__(self, id: int, 
               modelcfg: ModelConfig,
               outputfile: str="",
               save_latents: Literal["output"]="output") -> None:

    super().__init__(id, modelcfg, outputfile, save_latents)

  def load_data(self, *args, **kwargs) -> Dataset:
    dataset = load_from_disk(get_actual_path("smcq_dataset4_test", mode="data"))
    return dataset["test"]
  
  def select_data(self) -> Dataset:
    return super().select_data()

  def define_prompt(self, question: str, answer: str):
    return [
      {"role": "system", "content": QuestionRephrase.SYSTEM_PROMPT},
      *[item for sublist in [({"role": "user", "content": user}, {"role": "assistant", "content": assistant}) for user, assistant in QuestionRephrase.EXAMPLES] for item in sublist],
      {"role": "user", "content": f'Q: "{question}"\n'}
    ]

  def tokenize(self, tokenizer: PreTrainedTokenizer, prompt, examples) -> BatchEncoding:
    prompts = [tokenizer.apply_chat_template(prompt(question, answer), tokenize=False, add_generation_prompt=True) for question, answer in zip(examples["question"], examples["answer"])]
    return tokenizer(prompts, padding=True, return_tensors="pt", return_attention_mask=True)
  
  def batching(self, tokenized_dataset: Dataset, tokenizer) -> DataLoader:
    return DataLoader(tokenized_dataset, shuffle=True, batch_size=10,collate_fn=DataCollatorWithPadding(tokenizer))

  def result_buffer(self) -> pd.DataFrame:
    return pd.DataFrame(columns=[INDEX_COLUMN, "prompt", "reframed_question"])
  
  def save_latent_data(self, latent_data: pd.DataFrame):
    latent_data = latent_data.join(self.dataset.select_columns(["question", "answer", "label"]).to_pandas(), on=INDEX_COLUMN)
    return super().save_latent_data(latent_data)


  def compute_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
    return None
  
  def get_score(self, results) -> float:
    return None