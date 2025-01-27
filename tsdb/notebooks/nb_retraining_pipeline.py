# Databricks notebook source
# MAGIC %md
# MAGIC Future work:
# MAGIC
# MAGIC Parameterize objective metric and pruner

# COMMAND ----------

# MAGIC %run ./nb_config_retrieval

# COMMAND ----------

from functools import partial

import joblib
import mlflow
import optuna
from joblibspark import register_spark

from tsdb.ml.tune import objective
from tsdb.ml.promote import model_promotion
from tsdb.ml.yolo import YoloVersions
from tsdb.preprocessing.preprocess import build_mds_by_splits

# COMMAND ----------

# DBTITLE 1,Retrieve config file values
# Check if the global view exists
if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
    # Query the global temporary view and collect the first row
    result = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
    
    # Extract values from the result row
    env = result['env']
    catalog = result['catalog_name']
    schema = result['schema_name']
    table_name = result['gold_table_name']
    debug_mode = result['debug_mode'] == "true"
    unit_test_mode = result['unit_test_mode'] == "true"
    model_name = result['model_name']
else:
    # Exit the notebook with an error message if the global view does not exist
    dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

# DBTITLE 1,Create notebook parameters
# TODO: Add these parameters into the workflow under "parameters"
yolo_models = [member.name for member in YoloVersions] 
dbutils.widgets.dropdown("yolo_model", "yolov10n", yolo_models)

dbutils.widgets.dropdown("build_dataset", "False", ["True", "False"])
dbutils.widgets.dropdown("try_promotion", "False", ["True", "False"])
dbutils.widgets.text("num_trials", defaultValue="1")  
dbutils.widgets.text("table_version", defaultValue="397") 

# COMMAND ----------

# DBTITLE 1,Retrieve parameters from notebook
table_version = int(dbutils.widgets.get("table_version"))
yolo_model = dbutils.widgets.get("yolo_model")
num_trials = int(dbutils.widgets.get("num_trials"))
build_dataset = dbutils.widgets.get("build_dataset") == "True"
try_promotion = dbutils.widgets.get("try_promotion") == "True"

# COMMAND ----------

# DBTITLE 1,Build dataset from table if opted for
if build_dataset:
    location = "data/mds_training_splits" # make location a config parameter
    out_root_base_path = f"/Volumes/{catalog}/{schema}/" + location 

    table_version = build_mds_by_splits(
        catalog,
        schema,
        table_name,
        out_root_base_path
    )

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

out_root_base_path = f"/Volumes/{catalog}/{schema}/data/mds_training_splits/test_image_gold/version={table_version}"

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
objective_with_args = partial(
    objective,
    out_root_base=out_root_base_path,
    yolo_version="yolov10n",
    objective_metric="f1",
    model_name=model_name  # set model_name in config file?
)

# add with mlflow context here to get nested structure for logging
register_spark()

with mlflow.start_run():
    with joblib.parallel_backend("spark", n_jobs=-1):
        study.optimize(objective_with_args, n_trials=num_trials)

best_params = study.best_params

# COMMAND ----------

if try_promotion:
    
