# Databricks notebook source
# Purpose: Check if the global view 'global_temp_towerscout_configs' exists and extract configuration values from it. 
# If the view does not exist, exit the notebook with an error message.

# Check if the global view exists
if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
    # Query the global temporary view and collect the first row
    result = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
    
    # Extract values from the result row
    env = result['env']
    catalog = result['catalog_name']
    schema = result['schema_name']
    debug_mode = result['debug_mode'] == "true"
    unit_test_mode = result['unit_test_mode'] == "true"
else:
    # Exit the notebook with an error message if the global view does not exist
    dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

# MAGIC %pip install efficientnet_pytorch

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

from hyperopt import tpe, hp, SparkTrials

from functools import partial

from datetime import datetime

from petastorm.spark.spark_dataset_converter import SparkDatasetConverter

from tsdb.ml.train import perform_pass, train, tune_hyperparams, model_promotion
from tsdb.ml.utils import ValidMetric, TrainingArgs, FminArgs, SplitConverters, PromotionArgs 
from tsdb.utils.logger import setup_logger
from tsdb.utils.uc import CatalogInfo
from tsdb.ml.data_processing import split_datanolabel, get_converter_df

# COMMAND ----------

# project name folder
petastorm_path = "file:///dbfs/TowerScout/tmp/petastorm/dataloader_development_cache"

# Create petastorm cache
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, petastorm_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config file stuff

# COMMAND ----------

# widgets probs will go into config file
dbutils.widgets.text("source_schema", defaultValue="towerscout_test_schema")
dbutils.widgets.text("source_table", defaultValue="image_metadata")

dbutils.widgets.text("epochs", defaultValue="5")
dbutils.widgets.text("batch_size", defaultValue="1")
dbutils.widgets.text("report_interval", defaultValue="5")
dbutils.widgets.text("max_evals", defaultValue="8")
dbutils.widgets.text("parallelism", defaultValue="2")

stages = ["Dev", "Staging", "Production"]
dbutils.widgets.dropdown("stage", "Production", stages)

metrics = [member.name for member in ValidMetric]
dbutils.widgets.dropdown("objective_metric", "MSE", metrics)
dbutils.widgets.multiselect("metrics", "MSE", choices=metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook params

# COMMAND ----------

timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M"))
log_path = f"/Volumes/{catalog}/{schema}/test_volume/logs/towerscout_{timestamp}.log"
logger_name = "towerscout"
logger, handler = setup_logger(log_path, logger_name)
logger.info("This is an info message and the first message of the logs.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")

# COMMAND ----------

# DBTITLE 1,Retrieve parameters from notebook
objective_metric = dbutils.widgets.get("objective_metric")
epochs = int(dbutils.widgets.get("epochs"))
batch_size = int(dbutils.widgets.get("batch_size"))
report_interval = int(dbutils.widgets.get("report_interval"))
metrics = [ValidMetric[metric] for metric in dbutils.widgets.get("metrics").split(",")]
parallelism = int(dbutils.widgets.get("parallelism"))
max_evals = int(dbutils.widgets.get("max_evals"))
logger.info("Loaded parameters.")

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow client
client = MlflowClient()
logger.info("Created MLflow client.")

# COMMAND ----------

# DBTITLE 1,Data Ingestion and Splitting
source_table = dbutils.widgets.get("source_table")

table_name = f"{catalog}.{schema}.{source_table}"

images = spark.table(table_name).select("content", "path")

logger.info("Loading and splitting data into train, test, and validation sets")
train_set, test_set, val_set = split_datanolabel(images)

# COMMAND ----------

logger.info(f"Creating converter for train/val/test datasets")
# create converters for train/val/test spark df's
sc = spark.sparkContext
converter_train = get_converter_df(train_set, sc)
converter_val = get_converter_df(val_set, sc)
converter_test = get_converter_df(test_set, sc)

train_args = TrainingArgs(
    objective_metric=objective_metric,
    epochs=epochs,
    batch_size=batch_size,
    report_interval=report_interval,
    metrics=metrics,
)

split_convs = SplitConverters(
    train=converter_train, val=converter_val, test=converter_test
)

trials = SparkTrials(parallelism=parallelism)
logger.info(
    f"Training with {train_args.epochs} epochs, {train_args.batch_size} batch size, {train_args.report_interval} report interval, {train_args.objective_metric} objective metric"
)
search_space = {
    "lr": hp.uniform("lr", 1e-4, 1e-3),
    "weight_decay": hp.uniform("weight_decay", 1e-4, 1e-3),
}

# using partial to pass extra arugments to objective function
objective_func = partial(train, train_args=train_args, split_convs=split_convs)

fmin_args = FminArgs(
    fn=objective_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
)

# COMMAND ----------

try:
    best_run, challenger_test_metric, best_params = tune_hyperparams(
        fmin_args, train_args
    )
    logger.info(
        f"Best model has {train_args.objective_metric} of {challenger_test_metric}"
    )
    logger.debug("Hyperparameter tuning completed.")
except ValueError as e:
    logger.error(f"Invalid hyperparameter tuning arguments: {e}")
except RuntimeError as e:
    logger.error(f"Error during hyperparameter tuning: {e}")
except Exception as e:
    logger.error(f"Unexpected error during hyperparameter tuning: {e}")

# COMMAND ----------

logger.info(
    f"Testing metric ({train_args.objective_metric}) value of best run: {challenger_test_metric}"
)

# COMMAND ----------

# DBTITLE 1,Register challenger model
run_id = best_run.run_id
model_name = f"{catalog}.{schema}.towerscout_model"  # model name
alias = dbutils.widgets.get("stage")

challenger_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/ts-model-mlflow",  # path to logged artifact folder called models
    name=model_name,  # name for model in catalog
)
logger.info(
    f"Registered model {model_name} with version {challenger_model_metadata.version}"
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
    test_conv=split_convs.test,
    client=client,
    logger=logger,
)

# COMMAND ----------

try:
    model_promotion(promo_args)
    logger.debug("Promotion completed.")
except Exception as e:
    logger.error(f"Error during model promotion: {e}")

# COMMAND ----------

# DBTITLE 1,Delete converters
converter_train.delete()
converter_val.delete()
converter_test.delete()

# COMMAND ----------

# Close the handler to ensure the file is properly closed
handler.close()
logger.removeHandler(handler)
