from typing import TypeVar, Generic


T = TypeVar("T")


class Factory(Generic[T]):
    """Wraps a class into a factory object. When the object is called, an instance of
    the class is returned. Useful for when passing uninitialized objects between
    processes.
    
    The implementation logic is essentially:
    ```python
    def factory(cls, *args, **kwargs):
        def generate():
            return cls(*args, **kwargs)
        return generate
    ```"""

    def __init__(self, cls: T, *args, **kwargs):
        """
        Args:
            cls (T): Class or function to be called using `args` and `kwargs`.
                Additional args and kwargs may be added when called.
        """
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs) -> T:
        """Generates an object. Arguments and keyword arguments passed here are inserted
        __before__ those passed to the constructor.

        Returns:
            T: Object instance.
        """
        return self._cls(*args, *self._args, **kwargs, **self._kwargs)
