from typing import Any, final
from lightning.pytorch.callbacks import Callback


class BaseIntervalCallback(Callback):
    def __init__(self, every_n_epochs: int = None, every_n_steps: int = None):
        super().__init__()
        self.interval_epochs = every_n_epochs
        self.interval_steps = every_n_steps
        self._last_run_epoch = -1
        self._last_run_step = -1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Runtime enforcement
        forbidden = ['on_train_epoch_end', 'on_train_batch_end']
        for method in forbidden:
            if method in cls.__dict__:
                raise TypeError(
                    f"❌ Error: You attempted to override '{method}' in {cls.__name__}.\n"
                    f"   This method is managed by IntervalCallback.\n"
                    f"   Please override 'on_interval_{method.replace('on_', '')}' instead."
                )

    @final
    def on_train_epoch_end(self, trainer, pl_module):
        # Your interval logic here
        if self.interval_epochs and (trainer.current_epoch + 1) % self.interval_epochs == 0:
             if trainer.current_epoch != self._last_run_epoch:
                self.on_interval_train_epoch_end(trainer, pl_module)
                self._last_run_epoch = trainer.current_epoch

    def on_interval_train_epoch_end(self, trainer, pl_module):
        """Override this method to add your logic."""
        pass

    @final
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.interval_steps and (trainer.global_step + 1) % self.interval_steps == 0:
            if trainer.global_step != self._last_run_step:
                self.on_interval_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
                self._last_run_step = trainer.global_step

    def on_interval_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Override this method to add your logic."""
        pass


class IntervalWrapper(Callback):
    """
    Wraps any callback and makes it run at specified intervals.
    Preserves the original callback's state management.
    """
    def __init__(
        self, 
        wrapped_callback: Callback,
        every_n_epochs: int = None,
        every_n_steps: int = None
    ):
        super().__init__()
        self.wrapped = wrapped_callback
        self.interval_epochs = every_n_epochs
        self.interval_steps = every_n_steps
        self._last_run_epoch = -1
        self._last_run_step = -1

    @classmethod
    def of(
        cls,
        wrapped_callback_cls: type[Callback],
        every_n_epochs: int = None,
        every_n_steps: int = None,
        **kwargs,
    ):
        """Create a callback that runs the wrapped callback at specified intervals.

        When `every_n_epochs` is specified, the original callback's `on_train_epoch_end`
        will be called at the specified interval. When `every_n_steps` is specified, 
        the original callback's `on_train_batch_end` will be called at the specified interval.

        Args:
            wrapped_callback_cls: The class of the callback to wrap.
            every_n_epochs: The interval at which to run the wrapped callback on epoch end.
            every_n_steps: The interval at which to run the wrapped callback on batch end.
            **kwargs: Additional keyword arguments to pass to the wrapped callback class.

        Returns:
            A new callback that runs the wrapped callback at specified intervals.
        """
        return cls(
            wrapped_callback=wrapped_callback_cls(**kwargs),
            every_n_epochs=every_n_epochs,
            every_n_steps=every_n_steps,
        )
        
    # --- State Management ---
    
    def state_dict(self) -> dict[str, Any]:
        """Save both wrapper state AND wrapped callback's state."""
        return {
            '_last_run_epoch': self._last_run_epoch,
            '_last_run_step': self._last_run_step,
            'wrapped_state': self.wrapped.state_dict() if hasattr(self.wrapped, 'state_dict') else {}
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore both wrapper state AND wrapped callback's state."""
        self._last_run_epoch = state_dict.get('_last_run_epoch', -1)
        self._last_run_step = state_dict.get('_last_run_step', -1)
        
        # Restore wrapped callback's state if it exists
        if 'wrapped_state' in state_dict and hasattr(self.wrapped, 'load_state_dict'):
            self.wrapped.load_state_dict(state_dict['wrapped_state'])

    @property
    def state_key(self) -> str:
        # Include the wrapped callback's class name for uniqueness
        return self._generate_state_key(
            wrapped=self.wrapped.__class__.__name__,
            every_n_epochs=self.interval_epochs,
            every_n_steps=self.interval_steps
        )

    # --- Hook Delegation with Interval Logic ---
    
    def on_train_epoch_end(self, trainer, pl_module):
        should_run = False
        if self.interval_epochs:
            if (trainer.current_epoch + 1) % self.interval_epochs == 0:
                if trainer.current_epoch != self._last_run_epoch:
                    should_run = True
        
        if should_run:
            self.wrapped.on_train_epoch_end(trainer, pl_module)
            self._last_run_epoch = trainer.current_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        should_run = False
        if self.interval_steps:
            if (trainer.global_step + 1) % self.interval_steps == 0:
                if trainer.global_step != self._last_run_step:
                    should_run = True
        
        if should_run:
            self.wrapped.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            self._last_run_step = trainer.global_step

    # Delegate ALL other hooks to wrapped callback
    def __getattr__(self, name):
        """Forward any hook we didn't override to the wrapped callback."""
        return getattr(self.wrapped, name)
