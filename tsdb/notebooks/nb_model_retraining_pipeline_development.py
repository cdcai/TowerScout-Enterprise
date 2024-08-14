# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./demo_model

# COMMAND ----------

# MAGIC %run ./dataloader_development

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

from functools import partial # for passing extra arguements to obj func
from dataclasses import dataclass, asdict, field
from collections import namedtuple

from typing import Any

from enum import Enum

from torch import nn

# COMMAND ----------

# DBTITLE 1,Data Ingestion functions
def split_data(images: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """
    Splits a Spark dataframe into train, test, and validation sets.

    Args:
        df (DataFrame): Input dataframe to be split.
       

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    # split the dataframe into 3 sets
    images_train = images.sampleBy(("label"),fractions = {0: 0.8, 1: 0.8})
    images_remaining = images.join(images_train, on='path', how='leftanti') #remaining from images
    images_val = images_remaining.sampleBy(("label"),fractions = {0: 0.5, 1: 0.5}) # 50% of images_remaining
    images_test = images_remaining.join(images_val, on='path', how='leftanti') # remaining 50% from the images_remaining

    return images_train, images_test, images_val


def split_datanolabel(images: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """
    Splits a Spark dataframe into train, test, and validation sets.

    Args:
        df (DataFrame): Input dataframe to be split.
       

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    # split the dataframe into 3 sets
    images_train = images.sample(fraction = 0.8)
    images_remaining = images.join(images_train, on='path', how='leftanti') #remaining from images
    images_val = images_remaining.sample(fraction = 0.5) # 50% of images_remaining
    images_test = images_remaining.join(images_val, on='path', how='leftanti') # remaining 50% from the images_remaining

    return images_train, images_test, images_val

# COMMAND ----------

dbutils.widgets.text("source_schema", defaultValue="towerscout_test_schema")
dbutils.widgets.text("source_table", defaultValue="image_metadata")

dbutils.widgets.text("epochs", defaultValue="5")
dbutils.widgets.text("batch_size", defaultValue="10")
dbutils.widgets.text("report_interval", defaultValue="5")
dbutils.widgets.text("max_evals", defaultValue="16")

stages = ["Dev", "Staging", "Production"]
dbutils.widgets.dropdown("stage", "Production", stages)

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow client
client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Data Ingest
catalog_info = CatalogInfo.from_spark_config(spark) # CatalogInfo class defined in utils nb
catalog = catalog_info.name
schema = dbutils.widgets.get("source_schema")
source_table = dbutils.widgets.get("source_table")

table_name = f"{catalog}.{schema}.{source_table}"

images = (
    spark
    .table(table_name)
    .select("content", "path") 
    )
    
train_set, test_set, validation_set = split_datanolabel(images)

# COMMAND ----------

class ValidMetric(Enum):
    BCE = nn.BCEWithLogitsLoss()
    MSE = nn.MSELoss()

# COMMAND ----------

# If we setup multiselect and the optimization widget with this, we always guarantee the metric exists
# use .value to get actual function associated with member in ValidMetric
metrics = [member.name for member in ValidMetric]
dbutils.widgets.dropdown("objective_metric", "MSE", metrics)
dbutils.widgets.multiselect("metrics", "MSE" , choices=metrics)

# COMMAND ----------

# DBTITLE 1,Data classes and tuples
SplitDataset = namedtuple('SplitDataset', ['train', 'val', 'test'])
FminArgs = namedtuple('FminArgs', ['fn', 'space', 'algo', 'max_evals', 'trials'])

@dataclass
class TrainingArgs:
    """
    A class to represent model training arguements

    Attributes:
        objective_metric:
        epochs: 
        batch_size: 
        report_interval: 
        metrics: 
    """
    objective_metric: str = "recall" # will be selected option for the drop down
    epochs: int = 2
    batch_size: int = 4
    report_interval: int = 5
    metrics: list[ValidMetric] = field(default_factory=dict)

    @classmethod
    def from_dbutils_args(cls):
        # define class using notebook parameters
        obj_metric = dbutils.widgets.get("objective_metric")
        epochs = int(dbutils.widgets.get("epochs"))
        batch_size = int(dbutils.widgets.get("batch_size"))
        report_interval = int(dbutils.widgets.get("report_interval"))
        metrics = [ ValidMetric[metric] for metric in dbutils.widgets.get("metrics").split(",") ]
        return cls(obj_metric, epochs, batch_size, report_interval, metrics)

# COMMAND ----------

def get_converter(dataframe, context_args):
    dataframe = dataframe.transform(compute_bytes, "content")
    converter = create_converter(
        dataframe,
        "bytes"
    )
 
    return converter

def process_data(model_trainer, context_args, train_args, dataset, mode, epoch_num=0):
    metrics = {}
    converter = get_converter(dataset, context_args)
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

def train_tune_model(
        params: dict[str, Any], 
        train_args: TrainingArgs,
        split_data: SplitDataset
    ):
    """
    Trains a model with given hyperparameter values and returns the value of the evaluation metric on the valdiation dataset

    Args:
        params: dict[str, Any] The hyperparameter values to tune model with
        train_args: TrainingArgs The arguements for training and validaiton loops
        split_data SplitDataset The dataset split into train/val/test splits
    Returns:
        dict[str, float] A dict containing the loss 
    """

    with mlflow.start_run(nested=True):
        # Create model and trainer
        model_trainer = ModelTrainer(optimizer_args=params, metrics=train_args.metrics)
        mlflow.log_params(params)

        context_args = {
            "transform_spec": get_transform_spec(),
            "batch_size": train_args.batch_size
        } 
    
        # training
        for epoch in range(train_args.epochs):
            train_metrics = process_data(model_trainer, context_args, train_args, split_data.train, "TRAIN", epoch)
   
        # validation
        for epoch in range(train_args.epochs):
            val_metrics = process_data(model_trainer, context_args, train_args, split_data.val, "VAL", epoch) 

        # testing     
        test_metrics = process_data(model_trainer, context_args, train_args, split_data.test, "TEST")

        converter = get_converter(split_data.val, context_args)
        with converter.make_torch_dataloader(**context_args) as dataloader:
            dataloader_iter = iter(dataloader)
            images = next(dataloader_iter) # to get model signature
            # look into safetensors
            signature = infer_signature(model_input=images['features'].numpy(), \
                                        model_output=model_trainer.forward(images).logits.detach().numpy())
            mlflow.pytorch.log_model(model_trainer.model, "ts-model-mlflow", signature=signature)
        
        metric = val_metrics[f"{train_args.objective_metric}_VAL"] # minimize loss on val set b/c we are tuning hyperparams

    # Set the loss to -1*f1 so fmin maximizes the f1_score
    return {'status': STATUS_OK, 'loss': -1*metric}

# COMMAND ----------

def retrain(fmin_args: FminArgs, train_args: TrainingArgs):
    """
    Returns the best MLflow run and testing value of objective metric for that run

    Args:
        fmin_args: FminArgs The arguments to HyperOpt's fmin
        train_args: TrainingArgs The arguements for training and validaiton loops
    """
    with mlflow.start_run(run_name='towerscout_retrain'):
        best_params = fmin(**(fmin_args._asdict())) # cant pass raw namedtuple using **, must be mappable (dict)
      
    # sort by val objective_metric we minimize, using DESC so assuming higher is better
    best_run = mlflow.search_runs(order_by=[f'metrics.{train_args.objective_metric + "_VAL"} DESC']).iloc[0]
    # get test score of best run 
    best_run_test_metric = best_run[f"metrics.{train_args.objective_metric}_TEST"]
    print(f'Testing metric value of best run: {best_run_test_metric}')

    return best_run, best_run_test_metric, best_params

# COMMAND ----------

# investigate dbutils furhter, maybe use .getArguments vs .get?
# trials = SparkTrials(parallelism=4)
trials = Trials()

search_space = {
    'lr': hp.uniform('optlr', 1e-4, 1e-3), 
    'weight_decay': hp.uniform('optweight_decay', 1e-4, 1e-3)
    }

max_evals = int(dbutils.widgets.get("max_evals"))

# using dataclasses & namedtuples to shorten func signatures
train_args = TrainingArgs.from_dbutils_args()

split_data = SplitDataset(train=train_set, val=validation_set, test=test_set)

# using partial to pass extra arugments to objective function
objective_func = partial(
    train_tune_model, 
    train_args=train_args, 
    split_data=split_data
    )

fmin_args = FminArgs(
    fn=objective_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials
    )

best_run, contender_test_metric, best_params = retrain(fmin_args, train_args)

# COMMAND ----------

# DBTITLE 1,Register model
run_id = best_run.run_id
model_name = f"{catalog}.{schema}.towerscout_model"  # model name
alias = dbutils.widgets.get("stage")

contender_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/ts-model-mlflow", # path to logged artifact folder called models
    name=model_name # name for model in catalog
    )

# COMMAND ----------

# DBTITLE 1,Set registered model alias
client.set_registered_model_alias(
    name=model_name, 
    alias=alias, 
    version=1
)

# COMMAND ----------

# DBTITLE 1,Load registered model
prod_model = mlflow.pytorch.load_model(
          model_uri=f"models:/{model_name}@{alias}"
        )

# COMMAND ----------

# DBTITLE 1,Model Promotion Logic
# prod_model = mlflow.pyfunc.load_model(
#           model_uri=f"models:/{model_name}@{dbutils.widgets.get('stage')}"
#         )

# optimizer = set_optimizer(prod_model, **best_params) 
# model_trainer = TowerScoutModelTrainer(optimizer, train_args.metrics)

# context_args = {
#             "transform_spec": get_transform_spec(),
#             "batch_size": train_args.batch_size
#         }
# dataloader_iter, dataloader_size = get_dataloader(split_data.test, context_args)        

# test_metrics = process_data(model_trainer, dataloader_iter, dataloader_size, "test")
# current_best_test_metric = test_metrics[f"{train_args.objective_metric}/TEST"]

# contender_model_name = contender_model_metadata.name

# if contender_test_metric > current_best_test_metric:
#     print("Promoting contender model...")
#     client.set_registered_model_alias(
#         name=contender_model_name, 
#         alias=dbutils.widgets.get("stage"))
