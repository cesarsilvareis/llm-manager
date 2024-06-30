import re
from enum import Enum
from typing import Self, Type
from langchain.prompts import PromptTemplate

# Remediate type conflicts of the prompt hierarchy
class PromptType(Enum):
    RAW         = 0
    COMPLETION  = 1
    CONTROL     = 2
    PARAMETRIC  = 3

    def __str__(self) -> str:
        return self.name


class Prompt(str): # immutable & abstract

    # Avoiding prompt redundancy for multiple objects
    ACTIVE_PROMPTS: set[str] = set()

    def __new__(cls: Type[Self], content: str="", name: str="", *args) -> Self:
        cls.set_initialization(content, name, *args)
        return cls.initialize(content, *args)

    @classmethod
    def set_initialization(cls: Type[Self], content: str="", name: str="", *args):
        assert cls.check_format(content, name, *args)
        Prompt.ACTIVE_PROMPTS.add(name)

    @classmethod
    def initialize(cls: Type[Self], content: str="", *_) -> Self:
        return super(Prompt, cls).__new__(cls, content)


    @classmethod
    def check_format(cls: Type[Self], content: str, name: str, task:str) -> bool:
        return not(
            "" in [content, name, task] or \
            name in Prompt.ACTIVE_PROMPTS
        )

    def __init__(self: Self, _, name: str, type: PromptType, task: str):
        self._name = name   # unique identifier
        self._type = type   # non-shared type across descendency
        self._task = task

    @property
    def name(this: Self) -> str:
        return this._name
    
    @property
    def type(this: Self) -> PromptType:
        return this._type

    @property
    def task(this: Self) -> str:
        return this._task

    def __repr__(self: Self, **args) -> str:
        parameters = {
            "name": self.name,
            "type": self.type,
            "task": self.task,
            "content": repr(str(self)),
            **args
        }
        return (
            f"{self.__class__.__name__}(" +
            "; ".join(f"{k}='{v}'" for k, v in parameters.items())
            + ")"
        )
    
class RawPrompt(Prompt):

    def __init__(self: Self, _, name: str, task: str):
        super().__init__(_, name, PromptType.RAW, task)


class CompletionPrompt(Prompt):

    __NO_TASK__: str="NO_TASK"
    
    @classmethod
    def check_format(cls: Type[Self], content: str, name: str) -> bool:
        return super(CompletionPrompt, cls).check_format(content, name, cls.__NO_TASK__)
    
    def __init__(self: Self, _, name: str):
        super().__init__(_, name, PromptType.COMPLETION, CompletionPrompt.__NO_TASK__)


class SystemPrompt(Prompt):
    
    PATTERN = r"^System: {0}\n\nHuman: {1}$"

    def __new__(cls: type[Self], name: str, task: str, system_data: str, human_data: str) -> Self:
        return super().__new__(cls, f"System: {system_data}\n\nHuman: {human_data}", 
                               name, task, system_data, human_data)

    @classmethod
    def check_format(cls: Type[Self], content: str, name: str,
                     task: str, system_data: str, human_data: str) -> bool:
        if not(
            super(SystemPrompt, cls).check_format(content, name, task) and
            "" not in [system_data, human_data]
        ): return False
                    
        pattern = cls.PATTERN.format(system_data, human_data)
        regex = re.compile(pattern)
        return regex.match(content) is not None
    
    def __init__(self: Self, name: str, task: str, 
                 system_data: str, human_data: str):
        super().__init__(None, name, PromptType.CONTROL, task)
        self._system_data = system_data
        self._human_data = human_data

    @property
    def system_message(this: Self) -> str:
        return this._system_data
    
    @property
    def human_message(this: Self) -> str:
        return this._human_data
    

    def __repr__(self: Self) -> str:
        return super().__repr__(
            system=self.system_message,
            human=self.human_message
        )

    

class Template(Prompt):
    
    INSTANCE: PromptTemplate = None
    
    def __new__(cls: type[Self], content: str, name: str, task: str, **params: str) -> Self:
        return super().__new__(cls, content, name, task, params)

    @classmethod
    def check_format(cls: Type[Self], content: str, name: str, 
                     task: str, params: dict[str, str]) -> bool:
        
        if not super(Template, cls).check_format(content, name, task):
            return False
        
        t = cls.INSTANCE = PromptTemplate.from_template(content)    # keep reference
        if any(p not in params for p in t.input_variables):
            # So, we cannot build the template completely
            return False
        
        if any(p not in t.input_variables for p in params):
            # LOGS Warning
            ...

        return True
    
    @classmethod
    def initialize(cls: type[Self], content: str, _, params) -> Self:
        content = cls.INSTANCE.format(**params)
        return super().initialize(content)
    
    def __init__(self: Self, _, name: str, task: str, **params: str):
        super().__init__(_, name, PromptType.PARAMETRIC, task)
        self._params = params

    @property
    def params(this) -> dict[str, str]:
        return this._params

    def __repr__(self: Self) -> str:
        return super().__repr__(**self.params)
    

def main():
    p1 = f"""\
Make me this work:
    1. Wash dishes
    2. Remove trash
    3. Vacuum my house\
"""
    p2 = f"""\
The Road Not Taken

Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;\
"""

    p3_s = f"""\
You are a housekeeper with work already done for the day. \
However some kids spilled some chocolate milk in the floor. \
Assume the action given by the user.\
"""
    p3_h = f"""\
Do NOT let this pass you by! Go find a mop and clean the mess \
before the owner arives.\
"""
    
    p4 = f"""\
Find {{number_blogs}} best blogs about saving {{animal}}s \
on the Internet. For each one, make a list with the topics:
    1. {{topic_A}}
    2. {{topic_B}}
    3. {{topic_C}}
"""
    print("USER")
    raw_prompt = RawPrompt(p1, "p1", "Listing Me")
    completion_prompt = CompletionPrompt(p2, "p2")
    system_prompt = SystemPrompt("p3", "Cleaning my Home", p3_s, p3_h)
    template_prompt = Template(p4, "p4", "Save Animal Research", 
                               number_blogs="3",
                               animal="cat",
                               topic_A="Spatial dispersion",
                               topic_B="Food & Season",
                               topic_C="Nonprofit Organizations in Rescue and Welfare"
    )
    
    delimiter = f"\n{49*'#'}\n"
    print(raw_prompt, completion_prompt, system_prompt, template_prompt, sep=delimiter)
    print("DEV")
    print(delimiter.join(repr(p) for p in [raw_prompt, completion_prompt, system_prompt, template_prompt]))
    

if __name__ == "__main__":
    main()
