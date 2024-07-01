from pathlib import Path
from .utils import ImmutableMapping
from typing import Self


class ModelConfig(ImmutableMapping):
    def __init__(self, **kwargs):
        self._store = {
            **kwargs,
            'instances': {}
        }

    def save_instance(self, model, key: str):
        if key in self["instances"]:
            return
        
        self._store["instances"][key] = model

    def __getitem__(self, key):
        return self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._store})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ModelConfig) and other["name"] == self["name"]
    

class CurrentModel:

    LOCAL = "current-model"
    INSTANCE: 'CurrentModel' = None

    def __init__(self, modelcfg: ModelConfig) -> None:
        self._child = modelcfg

    @staticmethod
    def on():
        return CurrentModel.INSTANCE is not None

    @staticmethod
    def included(model: ModelConfig) -> bool:
        return CurrentModel.INSTANCE._child == model

    @staticmethod
    def initiate(child: ModelConfig):
        assert not CurrentModel.on()
        CurrentModel.INSTANCE = CurrentModel(child)

    @staticmethod
    def invalidate(new_modelcfg: ModelConfig):
        CurrentModel.INSTANCE.child["local"] = ""
        CurrentModel.INSTANCE.child = new_modelcfg
    