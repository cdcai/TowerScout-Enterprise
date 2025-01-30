from dataclasses import dataclass
from typing import Any

from optuna.trial import Trial


@dataclass
class Hyperparameters:
    """
    A class to represent hyperparameter arguements
    to be used for Optuna tuning

    Attributes:
        lr0: The initial learning rate
        momentum: optimizer momentum parameter
        weight_decay: optimizer weight decay parameter
        batch_size: batch size for dataloader
        epochs: number of epochs
        prob_H_flip: probablity of horizontally flipping image
        prob_V_flip: probability of vertically flipping image
        prob_mosaic: probablity of applying mosaic augmentation
    
    NOTE: Default values for parameters is taken from:
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    """

    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    batch_size: int = 16
    epochs: int = 100
    prob_H_flip: float = 0.5
    prob_V_flip: float = 0.0
    prob_mosaic: float = 1.0

    @classmethod
    def from_optuna_trial(cls, trial: Trial):  # pragma: no cover
        """
        NOTE: the class methods of Trial (suggest_int and suggest_float) used below to create the object are tested by Optuna:
        https://github.com/optuna/optuna/blob/46d7dab172ecb62ea3a0bad87e771ab25905f0ff/tests/trial_tests/test_trials.py
        """
        lr = trial.suggest_float("lr", 1e-2, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.937, 0.937)
        weight_decay = trial.suggest_float("weight_decay", 5e-4, 5e-4, log=True)
        batch_size_power = trial.suggest_int("batch_size_power", 4, 4)
        batch_size = 2**batch_size_power
        prob_H_flip = trial.suggest_float("prob_H_flip", 0.5, 0.5)
        prob_V_flip = trial.suggest_float("prob_V_flip", 0.0, 0.0)
        prob_mosaic = trial.suggest_float("prob_mosaic", 1.0, 1.0)
        epochs = trial.suggest_int("epochs", 1, 1)

        return cls(
            lr,
            momentum,
            weight_decay,
            batch_size,
            epochs,
            prob_H_flip,
            prob_V_flip,
            prob_mosaic,
        )


@dataclass
class TrainingArgs:
    """
    A class to represent model training arguements

    Attributes:
        objective_metric: The evaluation metric we want to optimize
        report_interval: Interval to log metrics during training
        val_interval: Interval to evaluate the model
        metrics: Various model evaluation metrics we want to track
        nbs: nominal batch size
        warmup_epochs: number of epochs to warmup the learning rate
        warmup_momentum: momentum for warmup
        warmup_bias_lr: bias learning rate for warmup
        lrf: final learning rate
        cos_lr: bool to use cosine annealing: https://arxiv.org/pdf/1812.01187
        patience: number of epochs to wait for validation metric improvement before early stopping
    """

    objective_metric: str = "BCE"  # will be selected option for the drop down
    report_interval: int = 5
    val_interval: int = 1
    metrics: Any = None
    nbs: int = 64  # nominal batch size
    warmup_epochs: float = 3.0  # fractions ok
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    lrf: float = 0.01
    cos_lr: bool = False  # maybe tune
    patience: int = 20