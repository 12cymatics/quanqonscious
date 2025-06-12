from typing import Any, Callable, Dict, List, Optional

import importlib

from .primarysutra import VedicSutras, SutraContext, SutraMode

class SutraRepository:
    """Lightweight wrapper exposing all sutras as callable functions."""

    def __init__(self, context: Optional[SutraContext] = None) -> None:
        self._sutras = VedicSutras(context)

        # Dynamically attach additional sutra functions from companion modules
        extra_modules = [
            'grvqsutraws',
            'intersutraws',
            'mayasutraaws',
            'sulbasutraws',
            'utilitysutraws3',
            'visualperformancesutraws2',
        ]
        for mod_name in extra_modules:
            try:
                mod = importlib.import_module('.' + mod_name, __package__)
            except Exception:
                continue
            for attr_name in dir(mod):
                if attr_name.startswith('_'):
                    continue
                func = getattr(mod, attr_name)
                if callable(func):
                    # Bind function to sutras instance so `self` works
                    setattr(self._sutras, attr_name, func.__get__(self._sutras))

        self._methods: Dict[str, Callable] = self._discover_sutras()

    def _discover_sutras(self) -> Dict[str, Callable]:
        """Collect all callable sutra methods from :class:`VedicSutras`."""
        methods: Dict[str, Callable] = {}
        for name in dir(self._sutras):
            if name.startswith("_"):
                continue
            attr = getattr(self._sutras, name)
            if callable(attr):
                methods[name] = attr
        return methods

    @property
    def context(self) -> SutraContext:
        return self._sutras.context

    def list_sutras(self) -> List[str]:
        """Return available sutra names."""
        return sorted(self._methods.keys())

    def call_sutra(self, name: str, *args, ctx: Optional[SutraContext] = None,
                   **kwargs) -> Any:
        """Invoke a sutra by name with the provided arguments.

        Parameters
        ----------
        name : str
            The sutra method to invoke.
        ctx : Optional[SutraContext], optional
            Execution context override. If omitted, the repository's current
            :class:`SutraContext` is used.
        *args, **kwargs :
            Arguments forwarded to the sutra implementation.
        """
        func = self._methods.get(name)
        if func is None:
            raise ValueError(f"Unknown sutra: {name}")

        # Ensure a context is passed if the sutra implementation accepts one
        if 'ctx' not in kwargs:
            kwargs['ctx'] = ctx or self.context

        return func(*args, **kwargs)

    def update_context(self, **kwargs: Any) -> None:
        """Update attributes of the underlying :class:`SutraContext`."""
        for key, value in kwargs.items():
            if hasattr(self._sutras.context, key):
                setattr(self._sutras.context, key, value)
