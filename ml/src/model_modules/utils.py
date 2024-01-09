from typing import Optional, Any

import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback

from utils.logging import get_logger

class MultiMSE:
    """MSE を各次元で別々に計算して各次元ごとに平均を保持する。"""

    sum_squared_error: Tensor | None
    total: int

    def __init__(self) -> None:
        self.reset()

    def update(self, preds: Tensor, target: Tensor) -> None:
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff, dim=0)

        if self.sum_squared_error is None:
            self.sum_squared_error = sum_squared_error
        else:
            self.sum_squared_error += sum_squared_error

        self.total += target.shape[0]

    def reset(self) -> None:
        self.sum_squared_error = None
        self.total: int = 0

    def compute(self) -> Tensor:
        return self.sum_squared_error / self.total

class LossErrorStopping(Callback):
    """loss が nan だったり inf だったり 0.0 だったりすると学習を停止させる"""
    def __init__(
        self,
        monitor: str,
        patience: int = 3,
        check_finite: bool = True,
        check_zero: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.check_finite = check_finite
        self.check_zero = check_zero
        self.wait_count = 0
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = None
        self.logger = get_logger(__name__)

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor, check_finite=self.check_finite, check_zero=self.check_zero
        )

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, logs: dict[str, Tensor]) -> bool:
        monitor_val = logs.get(self.monitor)
        if monitor_val is None:
            return False

        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.patience,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.patience = state_dict["patience"]

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn
        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking


    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)


    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason:
            self.logger.info(reason)

    def _evaluate_stopping_criteria(self, current: Tensor) -> tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = f"{self.monitor}: {current}, stop train"

        elif self.check_zero and not torch.any(current):
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = f"{self.monitor}: {current}, stop train"

        else:
            self.wait_count = 0

        return should_stop, reason
