from dataclasses import asdict
from typing import Any
from functools import partial

import mlflow
import torch
from optuna.trial import Trial
from torch.utils.data import DataLoader

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel

from tsdb.ml.utils import PromotionArgs, Hyperparameters, Steps
from tsdb.ml.data import DataLoaders, data_augmentation
from tsdb.ml.yolo_trainer import YoloModelTrainer
from tsdb.ml.model_trainer import TrainingArgs


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
    model.names = {0: "ct"}  # attach class names to model
    model.args = args
    # Note that this isn't set in cfg/default.yaml so must set it ourselves
    model.args.conf = 0.002
    # Set to true for towerscout since there's only 1 class
    model.args.single_cls = True

    return model


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
    model = get_model(f"{yolo_version}.yaml", f"{yolo_version}.pt")
    hyperparameters = Hyperparameters.from_optuna_trial(trial)
    train_args = TrainingArgs(objective_metric=objective_metric)
    model_trainer = YoloModelTrainer.from_optuna_hyperparameters(
        hyperparameters, model, train_args
    )

    transforms = data_augmentation(
        prob_H_flip=hyperparameters.prob_H_flip, prob_V_flip=hyperparameters.prob_V_flip
    )
    cache_dir = "/local/cache/path"

    dataloaders = DataLoaders.from_mds(
        cache_dir,
        mds_dir=out_root_base,
        hyperparams=hyperparameters,
        transforms=transforms,
    )

    with mlflow.start_run(nested=True):
        # Create model and trainer
        mlflow.log_params(asdict(hyperparameters))  # convert dataclass to dict
        metric = model_trainer.train(
            dataloaders, model_name="towerscout_model", trial=trial
        )

    return metric


def model_promotion(promo_args: PromotionArgs) -> None: # pragma: no cover
    # pass in args directly instead of dataclass
    """
    Evaluates the model that has the specficied alias. Promotes the model with the
    specfied alias

    Args:
        promo_args: A PromotionArgs dataclass containing the relevant arguments for 
                    model promotion logic
    Returns:
        None
    """
    mlflow.set_registry_uri("databricks-uc")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    challenger = mlflow.pytorch.load_model(promo_args.challenger_uri)
    challenger.to(device)

    validator = YoloModelTrainer.get_validator(
        dataloader=promo_args.testing_dataloader,
        training=False,
        device=device,
        args=challenger.args,
    )

    metrics = validator(challenger) 
    # b/c we use Ultralytics valdiator the metrics dict has keys "metrics/{metric_name}(B)"
    challenger_test_metric = metrics[f"metrics/{promo_args.comparison_metric}(B)"]

    # load current prod/champion model with matching alias
    champion = mlflow.pytorch.load_model(
        model_uri=f"models:/{promo_args.model_name}@{promo_args.alias}"
    )
    champion.to(device)

    metrics = validator(champion)
    champion_test_metric = metrics[f"metrics/{promo_args.comparison_metric}(B)"]

    if challenger_test_metric > champion_test_metric:
        print(f"Promoting challenger model to {promo_args.alias}.")
        
        # give alias to challenger model, alias is automatically removed from current champion model
        registration_info = mlflow.register_model(
            promo_args.challenger_uri, name=promo_args.model_name
        )
        client = mlflow.MlflowClient()
        promo_args.client.set_registered_model_alias(
            name=promo_args.model_name,
            alias=promo_args.alias,
            version=registration_info.version,  # version of challenger model from when it was registered
        )
    else:
        print(
            f"Challenger model does not perform better than current {promo_args.alias} model. Promotion aborted."
        )