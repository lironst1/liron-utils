import threading
import typing
from collections import OrderedDict
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Generic, TypeVar

import pandas as pd

_T = TypeVar("_T")


class MetaDict(type, Generic[_T]):
    """Metaclass exposing public class attributes as ``dict``-like items."""

    def __iter__(cls) -> Iterator[str]:
        for name in cls.__dict__:
            if not name.startswith("_"):
                yield name

    def __getitem__(cls, key: str) -> _T:
        return getattr(cls, key)  # type: ignore[no-any-return]

    def keys(cls) -> list[str]:
        return list(iter(cls))

    def values(cls) -> list[_T]:
        return [cls[k] for k in cls]  # pylint: disable=not-an-iterable

    def items(cls) -> list[tuple[str, _T]]:
        return [(k, cls[k]) for k in cls]  # pylint: disable=not-an-iterable


_K = TypeVar("_K")
_V = TypeVar("_V")


class dict_(dict[_K, _V]):  # pylint: disable=invalid-name
    """``dict`` subclass that exposes keys as attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]


def dl_to_ld(dl: dict[str, list[typing.Any]]) -> list[dict[str, typing.Any]] | None:
    """Convert a dict-of-lists into a list-of-dicts.

    Args:
        dl: Mapping ``{key: list-of-values}`` with equal-length lists.

    Returns:
        List of records (one dict per row), or ``None`` if ``dl`` values are scalars.
    """
    try:
        records = pd.DataFrame(dl).to_dict(orient="records")
    except ValueError:  # dict values are scalars
        return None
    return typing.cast(list[dict[str, typing.Any]], records)


def ld_to_dl(ld: list[dict[str, typing.Any]]) -> dict[str, list[typing.Any]]:
    """Convert a list-of-dicts into a dict-of-lists.

    Args:
        ld: Sequence of records sharing the same keys.

    Returns:
        Mapping ``{key: list-of-values}``.
    """
    dl = pd.DataFrame(ld).to_dict(orient="list")
    return typing.cast(dict[str, list[typing.Any]], dl)


class NamedQueue(Generic[_T]):
    """Thread-safe bounded FIFO whose items are addressable by name.

    Adding a name that already exists moves it to the most-recent position.
    When the queue is full, the oldest item is evicted to make room.
    """

    def __init__(self, max_size: int = 0) -> None:
        """Initialize the queue.

        Args:
            max_size: Maximum number of items; ``0`` means unbounded.
        """
        self.maxsize = max_size
        self.queue: OrderedDict[str, _T] = OrderedDict()
        self._lock = threading.Lock()

    def enqueue(self, name: str, item: _T) -> None:
        """Add ``item`` under ``name`` (evicts the oldest entry if full)."""
        with self._lock:
            if name in self.queue:
                del self.queue[name]
            elif 0 < self.maxsize <= len(self.queue):
                self.queue.popitem(last=False)  # FIFO order
            self.queue[name] = item

    def dequeue(self) -> tuple[str, _T] | None:
        """Pop and return the oldest ``(name, item)``, or ``None`` if empty."""
        with self._lock:
            if self.queue:
                return self.queue.popitem(last=False)
            return None

    def remove(self, name: str) -> _T:
        """Remove and return the item registered under ``name``."""
        with self._lock:
            return self.queue.pop(name)

    def update(self, name: str, item: _T) -> None:
        """Set the item registered under ``name``, inserting it if missing."""
        with self._lock:
            if name in self.queue:
                self.queue[name] = item
            else:
                self.enqueue(name, item)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self.queue

    def __len__(self) -> int:
        return len(self.queue)

    def __getitem__(self, name: str) -> _T:
        with self._lock:
            return self.queue[name]

    def __repr__(self) -> str:
        with self._lock:
            return f"Queue({list(self.queue.items())})"

    def __iter__(self) -> Iterator[tuple[str, _T]]:
        with self._lock:
            return iter(self.queue.items())

    def __next__(self) -> tuple[str, _T]:
        with self._lock:
            if self.queue:
                return self.queue.popitem(last=False)
            raise StopIteration

    def keys(self) -> KeysView[str]:
        with self._lock:
            return self.queue.keys()

    def values(self) -> ValuesView[_T]:
        with self._lock:
            return self.queue.values()

    def items(self) -> ItemsView[str, _T]:
        with self._lock:
            return self.queue.items()

    def clear(self) -> None:
        with self._lock:
            self.queue.clear()
