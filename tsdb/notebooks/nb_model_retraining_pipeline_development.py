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

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK, base
from hyperopt.pyll import scope

from functools import partial # for passing extra arguements to obj func
from dataclasses import dataclass, asdict, field
from collections import namedtuple

from typing import Any

from enum import Enum

from torch import nn

from petastorm.spark.spark_dataset_converter import SparkDatasetConverter

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

class ValidMetric(Enum):
    BCE = nn.BCEWithLogitsLoss()
    MSE = nn.MSELoss()

dbutils.widgets.text("source_schema", defaultValue="towerscout_test_schema")
dbutils.widgets.text("source_table", defaultValue="image_metadata")

dbutils.widgets.text("epochs", defaultValue="5")
dbutils.widgets.text("batch_size", defaultValue="10")
dbutils.widgets.text("report_interval", defaultValue="5")
dbutils.widgets.text("max_evals", defaultValue="16")
dbutils.widgets.text("parallelism", defaultValue="4")

stages = ["Dev", "Staging", "Production"]
dbutils.widgets.dropdown("stage", "Production", stages)

metrics = [member.name for member in ValidMetric]
dbutils.widgets.dropdown("objective_metric", "MSE", metrics)
dbutils.widgets.multiselect("metrics", "MSE" , choices=metrics)

# COMMAND ----------

# DBTITLE 1,Retrieve parameters from notebook
objective_metric = dbutils.widgets.get("objective_metric")
epochs = int(dbutils.widgets.get("epochs")) 
batch_size = int(dbutils.widgets.get("batch_size")) 
report_interval = int(dbutils.widgets.get("report_interval"))
metrics = [ValidMetric[metric] for metric in dbutils.widgets.get("metrics").split(",")]
parallelism = int(dbutils.widgets.get("parallelism"))
max_evals = int(dbutils.widgets.get("max_evals"))

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow client
client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Data Ingestion and Splitting
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
    
train_set, test_set, val_set = split_datanolabel(images)

# COMMAND ----------

print(train_set.count())
print(val_set.count())
print(test_set.count())

# COMMAND ----------

# DBTITLE 1,Data classes and named tuples
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

def get_converter_df(dataframe: DataFrame) -> callable:
    """
    Creates a petastrom converter for a Spark dataframe

    Args:
        dataframe: The Spark dataframe
    Returns:
        callable: A petastorm converter 
    """
    
    dataframe = dataframe.transform(compute_bytes, "content")
    converter = create_converter(
        dataframe,
        "bytes"
    )
 
    return converter

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

# COMMAND ----------

# MAGIC %md
# MAGIC Finish testing model promo logic
# MAGIC Add doc strings and type hints to funcs
# MAGIC Group funcs into seprate notebooks

# COMMAND ----------

# create converters for train/val/test spark df's
converter_train = get_converter_df(train_set)
converter_val = get_converter_df(val_set)
converter_test = get_converter_df(test_set)

train_args = TrainingArgs(
    objective_metric=objective_metric, 
    epochs=epochs, 
    batch_size=batch_size, 
    report_interval=report_interval, 
    metrics=metrics
    )

split_convs = SplitConverters(
    train=converter_train, 
    val=converter_val, 
    test=converter_test
    )

trials = SparkTrials(parallelism=parallelism)

search_space = {
    'lr': hp.uniform('lr', 1e-4, 1e-3), 
    'weight_decay': hp.uniform('weight_decay', 1e-4, 1e-3)
    }

# using partial to pass extra arugments to objective function
objective_func = partial(
    train, 
    train_args=train_args, 
    split_convs=split_convs
    )

fmin_args = FminArgs(
    fn=objective_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials
    )

# COMMAND ----------

best_run, challenger_test_metric, best_params = tune_hyperparams(fmin_args, train_args)

# COMMAND ----------

print(f'Testing metric ({train_args.objective_metric}) value of best run: {challenger_test_metric}')

# COMMAND ----------

# DBTITLE 1,Register challenger model
run_id = best_run.run_id
model_name = f"{catalog}.{schema}.towerscout_model"  # model name
alias = dbutils.widgets.get("stage")

challenger_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/ts-model-mlflow", # path to logged artifact folder called models
    name=model_name # name for model in catalog
    )

# COMMAND ----------

client.set_registered_model_alias(
        name=challenger_model_metadata.name, 
        alias=alias, 
        version=challenger_model_metadata.version # get version of challenger modelfrom when it was registered
    )

# COMMAND ----------

promo_args = PromotionArgs(
    objective_metric=train_args.objective_metric,
    batch_size=train_args.batch_size,
    metrics=train_args.metrics,
    model_version=challenger_model_metadata.version,
    model_name=model_name,
    challenger_metric_value=challenger_test_metric,
    alias=alias,
    test_conv=split_convs.test
    )

# COMMAND ----------

# DBTITLE 1,Model Promotion Logic
def model_promotion(promo_args: PromotionArgs) -> None:
    """
    Evaluates the model that has the specficied alias. Promotes the model with the
    specfied

    Args:
        promo_args: Contains arguments for the model promotion logic
    Returns:
        None
    """

    # load current model with matching alias (champion model)
    champ_model = mlflow.pytorch.load_model(
        model_uri=f"models:/{promo_args.model_name}@{promo_args.alias}"
        )
    
    #model_trainer = TowerScoutModelTrainer(optimizer_args={"lr": 0}, metrics=train_args.metrics)
    # use dummy optimzer params since optimizer is irrelevant for inference
    model_trainer = ModelTrainer(optimizer_args={"lr": 0}, metrics=promo_args.metrics)
    model_trainer.model = champ_model # Replace initial model with prod model

    context_args = {
                "transform_spec": get_transform_spec(),
                "batch_size": promo_args.batch_size
            }

    # get testing score for current produciton model
    steps_per_epoch = len(promo_args.test_conv) // promo_args.batch_size
    with promo_args.test_conv.make_torch_dataloader(**context_args) as dataloader:
        dataloader_iter = iter(dataloader)
        for minibatch_num in range(steps_per_epoch):
            minibatch_images = next(dataloader_iter)
            champ_model_test_metrics = model_trainer.validation_step(minibatch_images, "TEST")

    champ_test_metric = champ_model_test_metrics[f"{promo_args.objective_metric}_TEST"]
    print(f"{promo_args.objective_metric} for production model is: {champ_test_metric}")

    if challenger_test_metric > champ_test_metric:
        print(f"Promoting challenger model to {alias}.")
        # give alias to challenger model, alias is automatically removed from current champion model
        client.set_registered_model_alias(
            name=promo_args.model_name, 
            alias=promo_args.alias, 
            version=promo_args.model_version # version of challenger model from when it was registered
        )
    else:
        print("Challenger model performs worse than current production model. Promotion aborted.")
    

# COMMAND ----------

model_promotion(promo_args)

# COMMAND ----------

# DBTITLE 1,Delete converters
# converter_train.delete()
# converter_val.delete()
# converter_test.delete()
