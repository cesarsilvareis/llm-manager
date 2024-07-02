from src import get_actual_path
from src.logger import get_logger
from src.utils import BetterEnum, ImmutableMapping
from typing import Self

logger = get_logger(__name__)

class DownloadStatus(BetterEnum):
    UNINITIALIZED   =0
    STARTED         =1
    COMPLETED       =2

class ModelConfig(ImmutableMapping):
    def __init__(self, filename: str, **kwargs):
        self._filename = filename
        self._store = {**kwargs}
        self._instances = {}

    @property
    def filename(this) -> str:
        return this._filename
    
    @property
    def local(this) -> str:
        return this["local"] if this["local"] else CurrentModel.LOCAL

    @property
    def status(this) -> DownloadStatus:
        return this._store["download_status"]
    
    def _save(self):
        from src.loader import update_config
        update_config(self)

        logger.debug(f"Saved state for the model '{self['name']}'")


    def start_download(self):
        assert self.status == DownloadStatus.UNINITIALIZED
        self._store["download_status"] = str(DownloadStatus.STARTED)
        self._save()

    def invalidate_local(self):
        logger.debug(f"Invalidating local for model '{self['name']}'")
        self._store["download_status"] = str(DownloadStatus.UNINITIALIZED)
        self._save()

    def validate_local(self):
        logger.debug(f"Validating local for model '{self['name']}'")
        self._store["download_status"] = str(DownloadStatus.COMPLETED)
        self._save()

    def get_instance(self, key: str):
        return self._instances.get(key, None)

    def save_instance(self, model, key: str):
        if key in self._instances:
            return
        
        self._instances[key] = model

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
        logger.debug(f"Initializating current model as '{child['name']}'")

    @staticmethod
    def invalidate(new_modelcfg: ModelConfig):
        CurrentModel.INSTANCE._child.invalidate_local()
        CurrentModel.INSTANCE._child = new_modelcfg
    