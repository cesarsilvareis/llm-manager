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

    def __getitem__(self, key):
        return self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._store})"

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
    

####################
# Helper Functions #
####################


def return_all(f, all_arg: str, *args, return_as_tuple=False, **kwargs) -> Any|list[Any]:
    assert all_arg in kwargs

    if return_as_tuple:
        f = (lambda f: lambda *args, **kwargs: (kwargs[all_arg], f(*args, **kwargs)))(f)

    def call_f(s):
        kwargs[all_arg] = s
        return f(*args, **kwargs)
    
    all_value = kwargs[all_arg]
    if not isinstance(all_value, list):
        return call_f(all_value)

    return call_f(all_value[0]) if len(all_value) == 1 else [call_f(a) for a in all_value]
    