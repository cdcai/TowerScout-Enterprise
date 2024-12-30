import mlflow
from mlflow import MlflowClient

from optuna.trial import Trial

from dataclasses import asdict

from typing import Any, Callable, Union

from functools import partial

from enum import Enum

from torch import nn
from torch.utils.data import DataLoader

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel

from tsdb.ml.utils import TrainingArgs, FminArgs, PromotionArgs, OptimizerArgs
from tsdb.ml.data import DataLoaders
from tsdb.ml.data_processing import get_transform_spec
from tsdb.preprocessing.preprocess import build_mds_by_splits
from tsdb.ml.model_trainer import Steps, ModelTrainerType
from tsdb.ml.data import DataLoaders, data_augmentation
from tsdb.ml.yolo_trainer import inference_step, YoloModelTrainer
from tsdb.ml.utils import Steps


def perform_pass(
    step_func: Callable,
    dataloader: DataLoader,
    report_interval: int,
    epoch_num: int = 0,
) -> dict[str, float]:
    """
    Perfroms a single pass (epcoh) over the data accessed by the dataloader

    Args:
        step_func: A callable that performs either a training or inference step
        dataloader: The torch dataloader
        report_interval: How often to report metrics during pass
        epoch_num: The current epoch number for logging metrics across epochs
    Returns:
        dict[str, float] A dict containing values of various metrics for the epoch
    """

    metrics = {}
    num_batches = len(dataloader)

    for minibatch_num, minibatch in enumerate(dataloader):
        metrics = step_func(minibatch=minibatch)

        if minibatch_num % report_interval == 0:
            step_num = minibatch_num + (epoch_num * num_batches)
            mlflow.log_metrics(metrics, step=step_num)

    return metrics


def train(
    dataloaders: DataLoaders,
    model_trainer: ModelTrainerType,
    model_name: str = "towerscout_model",
) -> dict[str, Any]:  # pragma: no cover
    """
    Trains a model with given hyperparameter values and returns the value
    of the objective metric on the valdiation dataset.

    Args:
        model_trainer: The model trainer
        dataloaders: The dataloaders for the train/val/test datasets
        model_name: The name to log the model under in MLflow
    Returns:
        dict[str, float] A dict containing the loss
    """

    with mlflow.start_run(nested=True):
        # Create model and trainer
        mlflow.log_params(model_trainer.optimizer_args)

        train_args = model_trainer.train_args

        # training
        for epoch in range(train_args.epochs):
            train_metrics = perform_pass(
                step_func=model_trainer.training_step,
                dataloader=dataloaders.train,
                report_interval=train_args.report_interval,
                epoch_num=epoch,
            )

            if epoch % train_args.val_interval == 0:
                # validation
                val_metrics = perform_pass(
                    step_func=model_trainer.validation_step,
                    dataloader=dataloaders.val,
                    report_interval=len(dataloaders.val), # we want to run through whole validation dataloader and then log the metrics
                    epoch_num=epoch,
                )

        signature = model_trainer.get_signature(dataloaders.val)

        # Maybe we put this in the trainer?
        mlflow.pytorch.log_model(
            model_trainer.model,
            model_name,
            signature=signature,
        )

        metric = val_metrics[
            f"{train_args.objective_metric}_VAL"
        ]  # minimize loss on val set b/c we are tuning hyperparams

    return metric


def get_model(model_yaml: str, model_pt: str) -> DetectionModel:
    """
    Function for creating a DetectionModel object based on pretrained model weights
    and yaml file.
    See DetectionTrainer class and BaseTrainer class for details on how to setup the model

    Args:
        model_yaml: str, path to yaml file for YOLO model
        model_pt: str, path to pretrained YOLO model weights

    Returns:
        DetectionModel, an Ultralytics pytorch model with pretrained weights and class names attached
    """
    # get params for model from: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    args = get_cfg()
    model = DetectionModel(cfg=model_yaml, verbose=False)
    weights, _ = attempt_load_one_weight(model_pt)
    model.load(weights)
    model.nc = 1  # attach number of classes to model
    model.names = ["ct"]  # attach class names to model
    model.args = args
    # Note that this isn't set in cfg/default.yaml so must set it ourselves
    model.args.conf = 0.001
    # Set to true for towerscout since there's only 1 class
    model.args.single_cls = True

    return model


def objective(
    trial: Trial,
    out_root_base: str,
    yolo_version: str = "yolov10n",
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

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_args = OptimizerArgs("Adam", lr, momentum, weight_decay)
    # TODO: Make arguements to get_model inputs to this function
    model = get_model(f"{yolo_version}.yaml", f"{yolo_version}.pt")
    hyperparameters = 
    train_args = TrainingArgs(epochs=epochs) # pass optuna value for epochs

    model_trainer = YoloModelTrainer(optimizer_args, model, train_args)

    batch_size_power = trial.suggest_int("batch_size_power", 5, 10)
    batch_size = 2**batch_size_power
    prob_H_flip = trial.suggest_float("prob_H_flip", 0.3, 0.7)
    prob_V_flip = trial.suggest_float("prob_V_flip", 0.3, 0.7)

    transforms = data_augmentation(prob_H_flip=prob_H_flip, prob_V_flip=prob_V_flip)
    cache_dir = "/local/cache/path"

    dataloaders = DataLoaders.from_mds(
        cache_dir, mds_dir=out_root_base, batch_size=batch_size, transforms=transforms
    )
    
    with mlflow.start_run(nested=True):
        # Create model and trainer
        mlflow.log_params(hyperparameters)  # convert dataclass to dict
        metric = model_trainer.train(dataloaders, model_name="towerscout_model")

    return metric


def tune_hyperparams(
    fmin_args: FminArgs, train_args: TrainingArgs, run_name: str = "towerscout_retrain"
) -> tuple[Any, float, dict[str, Any]]:  # pragma: no cover
    """
    Returns the best MLflow run and testing value of objective metric for that run

    Optimize function

    Args:
        fmin_args: FminArgs The arguments to HyperOpts fmin function
        train_args: TrainingArgs The arguements for training and validaiton loops

    Returns:
        tuple[Any, float, dict[str, Any]] A tuple containing the best run, the value of the objective metric for that run, and the hyperparameters of that run and the assocaited best hyperparameters
    """
    with mlflow.start_run(run_name=run_name):
        best_params = fmin(**(fmin_args._asdict()))

    # sort by val objective_metric we minimize, using DESC so assuming higher is better
    best_run = mlflow.search_runs(
        order_by=[f'metrics.{train_args.objective_metric + "_VAL"} DESC']
    ).iloc[0]

    # get test score of best run
    best_run_test_metric = best_run[f"metrics.{train_args.objective_metric}_TEST"]
    mlflow.end_run()  # end run before exiting

    return best_run, best_run_test_metric, best_params


def model_promotion(promo_args: PromotionArgs) -> None:
    """
    Evaluates the model that has the specficied alias. Promotes the model with the
    specfied alias

    Args:
        promo_args: Contains arguments for the model promotion logic
    Returns:
        None
    """

    # load current model with matching alias (champion model)
    champ_model = mlflow.pytorch.load_model(
        model_uri=f"models:/{promo_args.model_name}@{promo_args.alias}"
    )

    # get testing score for current produciton model
    champ_model_test_metrics = perform_pass(
        step_func=partial(
            inference_step,
            model=champ_model,
            step=Steps["TEST"].name,
            metrics=promo_args.metrics,
        ),
        dataloader=promo_args.test_dataloader,
        report_interval=len(promo_args.test_dataloader),
    )

    champ_test_metric = champ_model_test_metrics[f"{promo_args.objective_metric}_TEST"]
    promo_args.logger.info(
        f"{promo_args.objective_metric} for production model is: {champ_test_metric}"
    )

    if promo_args.challenger_metric_value > champ_test_metric:
        promo_args.logger.info(f"Promoting challenger model to {promo_args.alias}.")
        # give alias to challenger model, alias is automatically removed from current champion model
        promo_args.client.set_registered_model_alias(
            name=promo_args.model_name,
            alias=promo_args.alias,
            version=promo_args.model_version,  # version of challenger model from when it was registered
        )
    else:
        promo_args.logger.info(
            f"Challenger model does not perform better than current {promo_args.alias} model. Promotion aborted."
        )