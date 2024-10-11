import re
from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict
from operator import itemgetter

from src.tasks import ED_MCQ
from src.training.data_preparation import EDStep2
from src.model import ModelConfig
from src import get_actual_path

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

class EDStep4(EDStep2): # Fix QA pairs

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
("""
Q: "Which of the following is a term describing a health education method comprising of a series of speeches on a selected subject?"
""", # Example 9: context + multi-answered
""" 
F: "What is the term that describes a health education method comprising of a series of speeches on a selected subject?"
"""),
("""
Q: "\"3-Million Plan\" was proposed by:-"
""", # Example 9: context + multi-answered
""" 
F: "Who proposed the \"3-Million Plan\"?"
"""),
  ]

  SYSTEM_PROMPT = """
You are given with the user's multiple-choice question (MCQ) potentially not \
logically connected to single answers when the universe of options is omitted. \
Your task is to remove its MCQ style (i.e., any "which of the following/below"-like \
text) to create a possible single-answer question. For that: provide all given \
necessary context (e.g., a clinical case); keep the meaning of the problem; \
and do NOT display the solution. 

In your generation, you must only present the fixed version \
of the question after the "F" mark. Any other thing should be avoided.
"""

  def __init__(self, task: ED_MCQ,
               dataset: Dataset|DatasetDict, # dataset 3
               modelcfg: ModelConfig, # llama2-7b
               split: str|None=None,
               ) -> None:
    super().__init__(task, dataset, split)

    from src.loader import load_model_from_hf
    load_model_from_hf(modelcfg)
    
    local = get_actual_path(modelcfg.local, "model")

    self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=local, **modelcfg["model_params"],
    ).eval()
    self.tokenizer = AutoTokenizer.from_pretrained(local, use_fast=False)
    
    self.tokenizer.padding_side = "left"
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model.resize_token_embeddings(len(self.tokenizer))
    self.model.config.pad_token_id = self.model.config.eos_token_id


  def define_prompt(self, question: str):
    return [
      {"role": "system", "content": self.SYSTEM_PROMPT},
      *[item for sublist in [({"role": "user", "content": user}, {"role": "assistant", "content": assistant}) for user, assistant in self.EXAMPLES] for item in sublist],
      {"role": "user", "content": f'Q: "{question}"\n'}
    ]

  def initialize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    def tokenize(examples):
      prompts = [self.tokenizer.apply_chat_template(self.define_prompt(question), tokenize=False, add_generation_prompt=True) for question in examples["question"][::2]]
      res_tokens = self.tokenizer(prompts, padding=True, return_tensors="pt", 
                                  return_attention_mask=True)
      return { k: list(i for i in l for _ in range(2)) for k, l in res_tokens.items() }
    
    tokenized = dataset.sort(self._task.QUESTION_LABEL).map(tokenize, batched=True, load_from_cache_file=False)
    
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized

  def fn_process(self, examples, _) -> Dataset:
    import torch

    terminators = [
      self.tokenizer.eos_token_id,
      self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
    batch = collator({ k: examples[k][::2] for k in ('input_ids', 'attention_mask') })
    input_ids = torch.tensor(batch['input_ids']).to(device=self.model.device)
    attention_mask = torch.tensor(batch['attention_mask']).to(device=self.model.device)

    outputs_ids = self.model.generate(input_ids, attention_mask=attention_mask,
                      num_return_sequences=1, eos_token_id=terminators, max_new_tokens=96, do_sample=True, temperature=0.25)

    fixed_questions = list(map(lambda i, o: self.tokenizer.decode(o[i.shape[-1]:],
                              skip_special_tokens=True), input_ids, outputs_ids))

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return {**examples, self._task.QUESTION_LABEL: list(q for q in fixed_questions for _ in range(2))}

  def finalize(self, dataset: Dataset, split: str | None = None) -> Dataset:
    r_qst_pat = re.compile(r'F:\s*"(.+)"$')

    def parse_question(example):
      return r_qst_pat.search(example["question"]) is not None

    return super().finalize(dataset, split)\
                  .filter(parse_question, batched=False)\
                  .map(lambda x: {**x, self._task.QUESTION_LABEL: r_qst_pat.search(x[self._task.QUESTION_LABEL]).group(1)}, batched=False)\
                  .shuffle(seed=self._task.SEED)

