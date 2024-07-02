from collections.abc import Mapping
from typing import Self, Any
from enum import Enum

class BetterEnum(Enum):
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self: Self, value: Any) -> bool:
        if isinstance(value, BetterEnum):
            return super().__eq__(self, value)
        
        return isinstance(value, str) and self.name.lower() == value.lower()


class ImmutableMapping(Mapping):

    def __setitem__(self, *_):
        raise TypeError(f"{self.__class__.__name__} object does not support item assignment")

    def __delitem__(self, _):
        raise TypeError(f"{self.__class__.__name__} object does not support item deletion")

    def update(self, *_, **__):
        raise TypeError(f"{self.__class__.__name__} object does not support update")

    def clear(self):
        raise TypeError(f"{self.__class__.__name__} object does not support clear")

    def pop(self, *_):
        raise TypeError(f"{self.__class__.__name__} object does not support pop")

    def popitem(self):
        raise TypeError(f"{self.__class__.__name__} object does not support popitem")