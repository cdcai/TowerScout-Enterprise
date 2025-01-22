# Databricks notebook source
# %run ./nb_config_retrieval

# COMMAND ----------

# # Purpose: Check if the global view 'global_temp_towerscout_configs' exists and extract configuration values from it. 
# # If the view does not exist, exit the notebook with an error message.

# # Check if the global view exists
# if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
#     # Query the global temporary view and collect the first row
#     result = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
    
#     # Extract values from the result row
#     env = result['env']
#     catalog = result['catalog_name']
#     schema = result['schema_name']
#     debug_mode = result['debug_mode'] == "true"
#     unit_test_mode = result['unit_test_mode'] == "true"
# else:
#     # Exit the notebook with an error message if the global view does not exist
#     dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

# widgets probs will go into config file
# dbutils.widgets.text("source_schema", defaultValue="towerscout_test_schema")  # config file
# dbutils.widgets.text("source_table", defaultValue="image_metadata")  # config file

# dbutils.widgets.text("report_interval", defaultValue="5")  # nb widget 
# dbutils.widgets.text("max_evals", defaultValue="8")  # nb widget
# dbutils.widgets.text("parallelism", defaultValue="2")  # nb widget

# metrics = [member.name for member in ValidMetric]
# dbutils.widgets.dropdown("objective_metric", "MSE", metrics)  # nb widget
# dbutils.widgets.multiselect("metrics", "MSE", choices=metrics)  # nb widget

# COMMAND ----------

# DBTITLE 1,Retrieve parameters from notebook
# objective_metric = dbutils.widgets.get("objective_metric")
# report_interval = int(dbutils.widgets.get("report_interval"))
# metrics = [ValidMetric[metric] for metric in dbutils.widgets.get("metrics").split(",")]
# parallelism = int(dbutils.widgets.get("parallelism"))
# max_evals = int(dbutils.widgets.get("max_evals"))

# COMMAND ----------

# from tsdb.preprocessing.preprocess import build_mds_by_splits

# out_root_base_path = "/Volumes/edav_dev_csels/towerscout/data/mds_training_splits"

# build_mds_by_splits(
#     "edav_dev_csels",
#     "towerscout",
#     "test_image_gold",
#     out_root_base_path
# )

# COMMAND ----------

# out_root_base_path = "/Volumes/edav_dev_csels/towerscout/data/mds_training_splits/test_image_gold/version=377"

# dataloaders = DataLoaders.from_mds(
#     cache_dir="/tmp/training_cache/", 
#     mds_dir=out_root_base_path, 
#     batch_size=32, 
#     transforms=data_augmentation()
# )

# COMMAND ----------

from functools import partial

import joblib
import mlflow
import optuna
from joblibspark import register_spark

from tsdb.ml.tune import objective

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

out_root_base_path = "/Volumes/edav_dev_csels/towerscout/data/mds_training_splits/test_image_gold/version=397"

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
objective_with_args = partial(
    objective,
    out_root_base=out_root_base_path,
    yolo_version="yolov10n",
    objective_metric="f1"
)

# add with mlflow context here to get nested structure for logging
register_spark()

with mlflow.start_run():
    with joblib.parallel_backend("spark", n_jobs=-1):
        study.optimize(objective_with_args, n_trials=1)

best_params = study.best_params

# COMMAND ----------

# try:
#     best_run, challenger_test_metric, best_params = tune_hyperparams(
#         fmin_args, train_args
#     )
#     logger.info(
#         f"Best model has {train_args.objective_metric} of {challenger_test_metric}"
#     )
#     logger.debug("Hyperparameter tuning completed.")
# except ValueError as e:
#     logger.error(f"Invalid hyperparameter tuning arguments: {e}")
# except RuntimeError as e:
#     logger.error(f"Error during hyperparameter tuning: {e}")
# except Exception as e:
#     logger.error(f"Unexpected error during hyperparameter tuning: {e}")

# COMMAND ----------

# logger.info(
#     f"Testing metric ({train_args.objective_metric}) value of best run: {challenger_test_metric}"
# )

# COMMAND ----------

# DBTITLE 1,Register challenger model
# run_id = best_run.run_id
# model_name = f"{catalog}.{schema}.towerscout_model"  # model name
# alias = dbutils.widgets.get("stage")

# challenger_model_metadata = mlflow.register_model(
#     model_uri=f"runs:/{run_id}/ts-model-mlflow",  # path to logged artifact folder called models
#     name=model_name,  # name for model in catalog
# )
# logger.info(
#     f"Registered model {model_name} with version {challenger_model_metadata.version}"
# )

# COMMAND ----------

# promo_args = PromotionArgs(
#     objective_metric=train_args.objective_metric,
#     batch_size=train_args.batch_size,
#     metrics=train_args.metrics,
#     model_version=challenger_model_metadata.version,
#     model_name=model_name,
#     challenger_metric_value=challenger_test_metric,
#     alias=alias,
#     test_conv=split_convs.test,
#     client=client,
#     logger=logger,
# )

# COMMAND ----------

# try:
#     model_promotion(promo_args)
#     logger.debug("Promotion completed.")
# except Exception as e:
#     logger.error(f"Error during model promotion: {e}")

# COMMAND ----------

# # Close the handler to ensure the file is properly closed
# handler.close()
# logger.removeHandler(handler)
