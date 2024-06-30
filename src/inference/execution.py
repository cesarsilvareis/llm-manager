from prompt import Prompt

class PromptExecution:

    def __init__(self, prompt: Prompt, model):
        self._prompt = prompt
        self._model = model

    @property
    def prompt(this) -> Prompt:
        return this._prompt
    
    @property
    def model(this):
        return this._model

    def __repr__(self) -> str:
        return (
            f"PromptExecution("
            f"prompt={self.prompt}; "
            f"model={self.model})"
        )