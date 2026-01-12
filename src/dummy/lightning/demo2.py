from typing import Callable, Type
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import functools
import logging

logger = logging.getLogger(__name__)


class VerboseCallbackWrapper:
    @staticmethod
    def wrap(_cls: Type[Callback]) -> Type[Callback]:
        """Wrap a callback class and return the wrapped class."""

        if not issubclass(_cls, Callback):
            logger.warning(f"{_cls} is not a subclass of Callback")

        hooks = [m for m in dir(Callback)
                 if m.startswith("on_") and callable(getattr(Callback, m))]

        if not hooks:
            raise ValueError(f"{_cls} has no on_* methods")

        class Wrapped(_cls):
            pass

        @rank_zero_only
        def func(epoch: int, orig: Callable):
            print(f"[Epoch {epoch}] Calling {orig.__name__}")

        def make_wrapper(orig, func: Callable):
            @functools.wraps(orig)
            def wrapper(self, *args, **kwargs):
                trainer = args[0] if args else None
                epoch = int(getattr(trainer, "current_epoch", -1))


                func(epoch, orig)

                return orig(self, *args, **kwargs)
            return wrapper

        for name in hooks:
            original = getattr(_cls, name, getattr(Callback, name))

            setattr(Wrapped, name, make_wrapper(original, func))

        return Wrapped


def verbose_callback(cls: Type[Callback]) -> Type[Callback]:
    return VerboseCallbackWrapper.wrap(cls)


@verbose_callback
class LoudCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(">>> Custom logic inside on_train_epoch_end")


class MockTrainer:
    current_epoch = 5

if __name__ == "__main__":
    callback = LoudCallback()

    trainer = MockTrainer()
    for _ in range(5):
        callback.on_train_epoch_end(trainer, None) # type: ignore
        trainer.current_epoch += 1

    print(f"{isinstance(callback, LoudCallback)}")