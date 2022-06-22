import abc
from typing import Any, Sequence, Union

__all__ = ["Storage", "ListStorage"]

from torchrl.data.replay_buffers.utils import INT_CLASSES


class Storage:
    @abc.abstractmethod
    def set(self, cursor: int, data: Any):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, index, value):
        return self.set(index, value)

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class ListStorage(Storage):
    def __init__(self):
        self._storage = []

    def set(self, cursor: Union[int, Sequence[int]], data: Any):
        if not isinstance(cursor, INT_CLASSES):
            for _cursor, _data in zip(cursor, data):
                self.set(_cursor, _data)
            return
        else:
            if cursor >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int]]) -> Any:
        if isinstance(index, int):
            return self._storage[index]
        else:
            return [self._storage[i] for i in index]

    def __len__(self):
        return len(self._storage)
