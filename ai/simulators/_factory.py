from typing import Generic, TypeVar


T = TypeVar("T")


class Factory(Generic[T]):
    """Factories are callable objects that spawn simulator instances."""

    def __init__(self, cls: T, *args, **kwargs):
        super().__init__()
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> T:
        return self._cls(*self._args, **self._kwargs)
