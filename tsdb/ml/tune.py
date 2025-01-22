from dataclasses import dataclass, asdict

from optuna.trial import Trial
import mlflow

from tsdb.ml.yolo import YoloModelTrainer
from tsdb.ml.datasets import DataLoaders
from tsdb.ml.types import TrainingArgs, Hyperparameters


def objective(
    trial: Trial,
    out_root_base: str,
    yolo_version: str = "yolov10n",
    objective_metric: str = "f1",
) -> float:  # pragma: no cover
    """
    Objective function for Optuna to optimize.

    Args:
        trail: Optuna Trail object for hyperparameter suggestions
        out_root_base: The directory to store the mds files
        yolo_version: the version of YOLO to use, default yolov10n
    Returns:
        The value of the objective metric to optimize after model trianing
        with suggested hyperparameters is completed
    """
    model =  YoloModelTrainer.get_model(f"{yolo_version}.yaml", f"{yolo_version}.pt")
    hyperparameters = Hyperparameters.from_optuna_trial(trial)
    train_args = TrainingArgs(objective_metric=objective_metric)
    model_trainer = YoloModelTrainer.from_optuna_hyperparameters(
        hyperparameters, model, train_args
    )

    cache_dir = "/local/cache/path"

    dataloaders = DataLoaders.from_mds(
        cache_dir,
        mds_dir=out_root_base,
        hyperparams=hyperparameters
    )

    with mlflow.start_run(nested=True):
        # Create model and trainer
        mlflow.log_params(asdict(hyperparameters))  # convert dataclass to dict
        metric = model_trainer.train(
            dataloaders, model_name="towerscout_model", trial=trial
        )

    return metric