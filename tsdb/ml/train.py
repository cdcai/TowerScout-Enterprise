import mlflow
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from hyperopt import fmin, STATUS_OK, SparkTrials, Trials

from dataclasses import dataclass, asdict, field
from collections import namedtuple

from typing import Any, Callable, Union

from functools import partial

from enum import Enum

from torch import nn

from tsdb.ml.utils import TrainingArgs, FminArgs, SplitConverters, PromotionArgs
from tsdb.ml.data_processing import get_transform_spec
from tsdb.ml.model_trainer import Steps, TowerScoutModelTrainer, inference_step


def perform_pass(
    step_func: Callable,
    converter: Callable,
    context_args: dict[str, Any],
    report_interval: int,
    epoch_num: int = 0,
) -> dict[str, float]:
    """
    Perfroms a single pass (epcoh) over the data accessed by the converter

    Args:
        step_func: A callable that performs either a training or inference step
        converter: The petastorm converter
        context_args: Arguments for the converter context
        report_interval: How often to report metrics during pass
        epoch_num: The current epoch number for logging metrics across epochs
    Returns:
        dict[str, float] A dict containing values of various metrics for the epoch
    """

    metrics = {}
    converter_length = len(converter)
    steps_per_epoch = converter_length // context_args["batch_size"]

    with converter.make_torch_dataloader(**context_args) as dataloader:
        dataloader_iter = iter(dataloader)
        
        for minibatch_num in range(steps_per_epoch):
            minibatch_images = next(dataloader_iter)
            metrics = step_func(minibatch=minibatch_images)
            
            if minibatch_num % report_interval == 0:
                step_num = minibatch_num + (epoch_num * converter_length)
                mlflow.log_metrics(
                    metrics,
                    step=step_num
                )
    
    return metrics


def train(
    params: dict[str, Any], train_args: TrainingArgs, split_convs: SplitConverters
) -> dict[str, Any]:  # pragma: no cover
    """
    Trains a model with given hyperparameter values and returns the value
    of the objective metric on the valdiation dataset.

    Args:
        params: The hyperparameter values to train model with
        train_args: The arguements for training and validaiton loops
        split_convs: The converters for the train/val/test datasets
    Returns:
        dict[str, float] A dict containing the loss
    """

    with mlflow.start_run(nested=True):
        # Create model and trainer
        model_trainer = TowerScoutModelTrainer(
            optimizer_args=params, 
            metrics=train_args.metrics
        )
        mlflow.log_params(params)

        context_args = {
            "transform_spec": get_transform_spec(),
            "batch_size": train_args.batch_size,
        }

        # training
        for epoch in range(train_args.epochs):
            train_metrics = perform_pass(
                step_func=model_trainer.training_step,
                converter=split_convs.train,
                context_args=context_args,
                report_interval=train_args.report_interval,
                epoch_num=epoch,
            )

        # validation
        for epoch in range(train_args.epochs):
            val_metrics = perform_pass(
                step_func=model_trainer.validation_step,
                converter=split_convs.val,
                context_args=context_args,
                report_interval=len(split_convs.val),
                epoch_num=epoch,
            )

        # testing
        test_metrics = perform_pass(
            step_func=partial(inference_step, model=model_trainer.model, step=Steps["TEST"].name, metrics=train_args.metrics), 
            converter=split_convs.test, 
            context_args=context_args, 
            report_interval=len(split_convs.test)
        )

        with split_convs.test.make_torch_dataloader(**context_args) as dataloader:
            dataloader_iter = iter(dataloader)

            images = next(dataloader_iter)  # to get model signature

            signature = infer_signature(
                model_input=images["features"].numpy(),
                model_output=model_trainer.model(images["features"]).detach().numpy(),
            )

            mlflow.pytorch.log_model(
                model_trainer.model, "ts-model-mlflow", signature=signature
            )

        metric = val_metrics[
            f"{train_args.objective_metric}_VAL"
        ]  # minimize loss on val set b/c we are tuning hyperparams

    # Set the loss to -1*f1 so fmin maximizes the f1_score
    return {"status": STATUS_OK, "loss": -1 * metric}


def tune_hyperparams(
    fmin_args: FminArgs, train_args: TrainingArgs
) -> tuple[Any, float, dict[str, Any]]:  # pragma: no cover
    """
    Returns the best MLflow run and testing value of objective metric for that run

    Args:
        fmin_args: FminArgs The arguments to HyperOpts fmin function
        train_args: TrainingArgs The arguements for training and validaiton loops

    Returns:
        tuple[Any, float, dict[str, Any]] A tuple containing the best run, the value of the objective metric for that run, and the hyperparameters of that run and the assocaited best hyperparameters
    """
    with mlflow.start_run(run_name="towerscout_retrain"):
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

    context_args = {
        "transform_spec": get_transform_spec(),
        "batch_size": promo_args.batch_size,
    }

    # get testing score for current produciton model
    champ_model_test_metrics = perform_pass(
        step_func=partial(inference_step, model=champ_model, step=Steps["TEST"].name, metrics=promo_args.metrics),
        converter=promo_args.test_conv,
        context_args=context_args,
        report_interval=len(promo_args.test_conv)
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
