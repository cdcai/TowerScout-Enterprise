# Databricks notebook source
import mlflow
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK, base
from hyperopt.pyll import scope

from dataclasses import dataclass, asdict, field
from collections import namedtuple

from typing import Any

from enum import Enum

from petastorm.spark.spark_dataset_converter import SparkDatasetConverter

# COMMAND ----------

class ValidMetric(Enum):
    """
    An Enum which is used to represent valid evaluation metrics for the model
    """
    BCE = nn.BCEWithLogitsLoss()
    MSE = nn.MSELoss()

# COMMAND ----------

FminArgs = namedtuple('FminArgs', ['fn', 'space', 'algo', 'max_evals', 'trials'])

@dataclass
class SplitConverters:
    """
    A class to hold the spark dataset converters for the training, testing
    and validation sets 

    Attributes:
        train: The spark dataset converter for the training dataset
        val: The spark dataset converter for the validation dataset
        test: The spark dataset converter for the testing dataset
    """
    train: SparkDatasetConverter = None
    val: SparkDatasetConverter = None
    test: SparkDatasetConverter = None

@dataclass
class TrainingArgs:
    """
    A class to represent model training arguements

    Attributes:
        objective_metric:The evaluation metric we want to optimize
        epochs: Number of epochs to optimize model over
        batch_size: The size of the minibatchs passed to the model
        report_interval: Interval to log metrics during training
        metrics: Various model evaluation metrics we want to track 
    """
    objective_metric: str = "recall" # will be selected option for the drop down
    epochs: int = 2
    batch_size: int = 4
    report_interval: int = 5
    metrics: list[ValidMetric] = field(default_factory=dict)

@dataclass
class PromotionArgs:
    """
    A class to represent model promotion arguements

    Attributes:
        objective_metric: The evaluation metric we want to optimize
        batch_size: The size of the minibatchs passed to the model
        metrics: Various model evaluation metrics we want to track 
        model_version: The version of the model that is the challenger
        model_name: The name of the model 
        challenger_metric_value: The value of the objective metric achieved by the challenger model on the test dataset
        alias: The alias we are promoting the model to
        test_conv: The converter for the test dataset   
    """
    objective_metric: str = "recall"
    batch_size: int = 4
    metrics: list[ValidMetric] = field(default_factory=dict)
    model_version: int = 1
    model_name: str = "ts"
    challenger_metric_value: float = 0
    alias: str = "staging"
    test_conv: SparkDatasetConverter = None

# COMMAND ----------

def perform_pass(
        model_trainer: ModelTrainer, 
        converter: callable, 
        context_args: dict[str, Any], 
        train_args: TrainingArgs, 
        mode: str, 
        epoch_num: int = 0
    ) -> dict[str, float]:
    """
    Perfroms a single pass (epcoh) over the data accessed by the converter

    Args:
        model_trainer: The model trainer
        converter: The petastorm converter 
        context_args: Arguments for the converter context
        train_args: Contains training arguments such as batch size
        mode: Specifics if model is in training or evalaution mode
        epoch_num: The current epoch number for logging metrics across epochs
    Returns:
        dict[str, float] A dict containing values of various metrics for the epoch
    """

    metrics = {}
    converter_length = len(converter)
    steps_per_epoch = converter_length // train_args.batch_size
    if mode == "TRAIN":
       report_interval = train_args.report_interval
    else:
        report_interval = converter_length

    with converter.make_torch_dataloader(**context_args) as dataloader:
        dataloader_iter = iter(dataloader)
        for minibatch_num in range(steps_per_epoch):
            minibatch_images = next(dataloader_iter)
            if mode == "TRAIN":
                metrics = model_trainer.training_step(minibatch_images, mode)
            else:
                metrics = model_trainer.validation_step(minibatch_images, mode)
            if minibatch_num % report_interval == 0:
                is_train = mode == "TRAIN"
                mlflow.log_metrics(metrics, step=is_train*(minibatch_num + epoch_num*converter_length))

    return metrics

# COMMAND ----------

def train(
        params: dict[str, Any], 
        train_args: TrainingArgs,
        split_convs: SplitConverters
    ) -> dict[str, Any]:
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
        #model_trainer = TowerScoutModelTrainer(optimizer_args=params, metrics=train_args.metrics)
        model_trainer = ModelTrainer(optimizer_args=params, metrics=train_args.metrics)
        mlflow.log_params(params)
        
        context_args = {
            "transform_spec": get_transform_spec(),
            "batch_size": train_args.batch_size
        }
        
        # training
        for epoch in range(train_args.epochs):
            train_metrics = perform_pass(model_trainer, split_convs.train, context_args, train_args, "TRAIN", epoch)
   
        # validation
        for epoch in range(train_args.epochs):
            val_metrics = perform_pass(model_trainer, split_convs.val, context_args, train_args, "VAL", epoch) 

        # testing     
        test_metrics = perform_pass(model_trainer, split_convs.test, context_args, train_args, "TEST")

        
        with split_convs.test.make_torch_dataloader(**context_args) as dataloader:
            dataloader_iter = iter(dataloader)
            
            images = next(dataloader_iter) # to get model signature
            
            signature = infer_signature(model_input=images['features'].numpy(), 
                                        model_output=model_trainer.forward(images).logits.detach().numpy())
            
            mlflow.pytorch.log_model(model_trainer.model, "ts-model-mlflow", signature=signature)
        
        metric = val_metrics[f"{train_args.objective_metric}_VAL"] # minimize loss on val set b/c we are tuning hyperparams

    # Set the loss to -1*f1 so fmin maximizes the f1_score
    return {'status': STATUS_OK, 'loss': -1*metric}

# COMMAND ----------

def tune_hyperparams(
                    fmin_args: FminArgs, 
                     train_args: TrainingArgs
    ) -> tuple[Any, float, dict[str, Any]]:
    """
    Returns the best MLflow run and testing value of objective metric for that run

    Args:
        fmin_args: FminArgs The arguments to HyperOpts fmin function
        train_args: TrainingArgs The arguements for training and validaiton loops
    """
    with mlflow.start_run(run_name='towerscout_retrain'):
        best_params = fmin(**(fmin_args._asdict())) # cant pass raw dataclass using **, must be mappable (dict)
      
    # sort by val objective_metric we minimize, using DESC so assuming higher is better
    best_run = mlflow.search_runs(order_by=[f'metrics.{train_args.objective_metric + "_VAL"} DESC']).iloc[0]
    
    # get test score of best run 
    best_run_test_metric = best_run[f"metrics.{train_args.objective_metric}_TEST"]
    mlflow.end_run() # end run before exiting

    return best_run, best_run_test_metric, best_params
