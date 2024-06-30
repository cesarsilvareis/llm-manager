from pathlib import Path
from .utils import ImmutableMapping


class ModelConfig(ImmutableMapping):
    def __init__(self, name: str, hf_repo: str, revision_hash: str, local: Path, **params):
        self._store = {
            'name': name,
            'hf_repo': hf_repo,
            'revision_hash': revision_hash,
            'local': local,
            'params': params
        }

    def __getitem__(self, key):
        return self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._store})"



    

    