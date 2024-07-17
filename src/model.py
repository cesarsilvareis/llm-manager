from src import get_actual_path
from src.logger import get_logger
from src.utils import BetterEnum, ImmutableMapping
from typing import Self, Any
from datetime import datetime

logger = get_logger(__name__)

class DownloadStatus(BetterEnum):
    UNINITIALIZED   =0
    STARTED         =1
    COMPLETED       =2

class ModelConfig(ImmutableMapping):

    def __init__(self, filename: str, **kwargs):
        self._filename = filename
        self._store = {**kwargs}
        self._instance = None

        self._store["experiment_date"] = str(datetime.now())

    @property
    def filename(this) -> str:
        return this._filename
    
    @property
    def local(this) -> str:
        return this["local"] if this["local"] else CurrentModel.LOCAL

    @property
    def status(this) -> DownloadStatus:
        return this._store["download_status"]
    
    @property
    def instance(this):
        return this._instance # DO NOT NEED THIS! REFERENCES!!!

    
    def _savecfg(self):
        from src.loader import update_config
        update_config(self)

        logger.debug(f"Saved state for the model '{self['name']}'")

    def start_download(self):
        assert self.status == DownloadStatus.UNINITIALIZED
        self._store["download_status"] = str(DownloadStatus.STARTED)
        self._savecfg()

    def invalidate_local(self):
        logger.debug(f"Invalidating local for model '{self['name']}'")
        self._store["download_status"] = str(DownloadStatus.UNINITIALIZED)
        self._savecfg()

    def validate_local(self):
        logger.debug(f"Validating local for model '{self['name']}'")
        self._store["download_status"] = str(DownloadStatus.COMPLETED)
        self._savecfg()

    def load_instance(self, caller, *args, **kwargs): # insecured by design ;)
        if self.instance is not None: return

        from src.loader import load_model_from_hf
        load_model_from_hf(self)

        logger.debug(f"Loading model '{self['name']}' with arguments: {args=}; {kwargs=}")
        self._instance = caller(*args, **kwargs)
        
        # logger.info(f"Loaded model '{self['name']}' with  footprint: {(self.instance.model.get_memory_footprint()/1e9):.3f} GB of (V)RAM")

    def teardown(self):
        import torch
        self._instance = None
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # keeping the model save awaiting for replacement! 

    def __hash__(self):
        return hash(self["name"])

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
    