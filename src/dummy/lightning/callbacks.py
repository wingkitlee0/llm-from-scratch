import functools
import inspect
from typing import Callable

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


def verbose_callback(cls):
    """
    Class decorator that wraps all `on_` methods of a Lightning Callback
    to print a log message before execution.
    """
    # Get all standard hook names from the base Callback class
    base_hooks = [
        m
        for m in dir(Callback)
        if m.startswith("on_") and inspect.isfunction(getattr(Callback, m))
    ]

    def create_wrapper(original_method: Callable) -> Callable:
        @functools.wraps(original_method)
        def wrapper(self, *args, **kwargs):
            # Attempt to extract trainer to get the current epoch
            trainer = args[0] if args else None
            epoch = getattr(trainer, "current_epoch", "?")

            # Use rank_zero_only to prevent spam in distributed training (DDP)
            @rank_zero_only
            def _log():
                method_name = original_method.__name__
                print(f"[Epoch {epoch}] Calling {method_name}")

            _log()

            # Call the original method (or the user's override)
            return original_method(self, *args, **kwargs)

        return wrapper

    for name in base_hooks:
        # Get the implementation to wrap:
        # either the user's override in `cls` or the base method from `Callback`
        original_impl = getattr(cls, name, getattr(Callback, name))

        # Wrap and replace the method on the class
        setattr(cls, name, create_wrapper(original_impl))

    return cls


# --- Usage ---


@verbose_callback
class LoudCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # You can override methods normally.
        # The decorator wraps this method, so it prints AND runs your logic.
        print(">>> Custom logic inside on_train_epoch_end")
