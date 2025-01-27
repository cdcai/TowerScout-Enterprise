# Databricks notebook source
# MAGIC %md
# MAGIC Future work:
# MAGIC
# MAGIC Parameterize objective metric and pruner

# COMMAND ----------

# MAGIC %run ./nb_config_retrieval

# COMMAND ----------

from functools import partial
from uuid import uuid4

import joblib
import mlflow
import optuna
from joblibspark import register_spark

from tsdb.ml.tune import objective
from tsdb.ml.promote import model_promotion
from tsdb.ml.yolo import YoloVersions, EvaluationMetrics
from tsdb.ml.utils import UCModelName
from tsdb.ml.datasets import get_dataloader
from tsdb.ml.types import Hyperparameters
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
    gold_table_name = result['gold_table_name']
    debug_mode = result['debug_mode'] == "true"
    unit_test_mode = result['unit_test_mode'] == "true"
    model_name = result['model_name']
    mds_path = result['mds_path']
    alias = result['champion_alias']
else:
    # Exit the notebook with an error message if the global view does not exist
    dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

# DBTITLE 1,Create notebook parameters
# TODO: Add these parameters into the workflow under "parameters"
yolo_models = [member.name for member in YoloVersions] 
dbutils.widgets.dropdown("yolo_model", "yolov10n", yolo_models)

eval_metrics = [member.value for member in EvaluationMetrics] 
dbutils.widgets.dropdown("objective_metric", "f1", eval_metrics)

dbutils.widgets.dropdown("build_dataset", "False", ["True", "False"])
dbutils.widgets.dropdown("try_promotion", "False", ["True", "False"])
dbutils.widgets.text("num_trials", defaultValue="1")  
dbutils.widgets.text("table_version", defaultValue="397") 

# COMMAND ----------

# DBTITLE 1,Retrieve parameters from notebook
table_version = int(dbutils.widgets.get("table_version"))
yolo_model = dbutils.widgets.get("yolo_model")
objective_metric = dbutils.widgets.get("objective_metric")
num_trials = int(dbutils.widgets.get("num_trials"))
build_dataset = dbutils.widgets.get("build_dataset") == "True"
try_promotion = dbutils.widgets.get("try_promotion") == "True"

# COMMAND ----------

# DBTITLE 1,Build dataset from table if opted for
if build_dataset:
    out_root_base_path = f"/Volumes/{catalog}/{schema}/{mds_path}"

    table_version = build_mds_by_splits(
        catalog,
        schema,
        gold_table_name,
        out_root_base_path
    )

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

out_root_base_path = f"/Volumes/{catalog}/{schema}/{mds_path}/{gold_table_name}/version={table_version}"

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
objective_with_args = partial(
    objective,
    out_root_base=out_root_base_path,
    yolo_version=yolo_model,
    objective_metric=objective_metric,
    model_name=f"{model_name}_{yolo_model}"
)

register_spark()

# add with mlflow context here to get nested structure for logging
with mlflow.start_run():
    with joblib.parallel_backend("spark", n_jobs=-1):
        study.optimize(objective_with_args, n_trials=num_trials)

best_params = study.best_params
best_trial = study.best_trial

# COMMAND ----------

if try_promotion:
    hyperparams = Hyperparameters() 

    testing_dataloader = get_dataloader(
        local_dir=f"/local/cache/path/{str(uuid4())}",
        remote_dir=f"{out_root_base_path}/test",
        hyperparams=hyperparams,
        transform=False,
    )

    uc_model_name = UCModelName(catalog, schema, model_name)

    model_promotion(
        challenger_uri=best_trial.user_attrs["model_uri"],
        testing_dataloader=testing_dataloader,
        comparison_metric=objective_metric,
        uc_model_name=uc_model_name,
        alias=alias,
    )
