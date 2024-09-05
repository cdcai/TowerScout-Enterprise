# Databricks notebook source
# MAGIC %md
# MAGIC # NOTE
# MAGIC These notebooks aren't designed to play well with more than one person running them. Rather than to account for that, we use these as guiding examples to demonstrate Spark's capabilities. Be mindful of cells that write or delete content.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import load_wine

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

import pandas as pd
import seaborn as sns

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow clieant
client = MlflowClient()

# COMMAND ----------

# load diabetes data
dataset = load_wine(as_frame=True)
dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
dataset_df['target'] = pd.Series(dataset.target)

display(dataset_df[:5])

# COMMAND ----------

sns.displot(dataset_df.target, kde=False)

# COMMAND ----------

target_col = "target" # quantitative measure of disease progression one year after baseline
X = dataset_df.drop(columns=[target_col])
y = dataset_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=22)

# COMMAND ----------

# MAGIC %md
# MAGIC # Baseline mode

# COMMAND ----------


def train_model(X_train, y_train):
    # Model parameters 
    n_estimators = 2
    max_depth = 2
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
  
    model.fit(X_train, y_train)
    
    # log hyperparameters
    mlflow.log_param("max_depth", max_depth)  
    mlflow.log_param("n_estimators", n_estimators)

    # log training error metric
    f1 = f1_score(model.predict(X_train), y_train, average="weighted")
    mlflow.log_metric("my_f1_training", f1)

    return model

mlflow.sklearn.autolog(disable=True) # disable autologging for baseline model

with mlflow.start_run(run_name='baseline_rf_model') as run:
  # Create and train model
  model = train_model(X_train, y_train)

  # Inference on test data
  y_pred = model.predict(X_test)

  # get model signature (input and output schema)
  signature = infer_signature(X_test, y_pred)

  # log testing error metric
  f1 = f1_score(y_pred, y_test, average="weighted")
  mlflow.log_metric("my_f1_testing", f1)

  # log the model
  mlflow.sklearn.log_model(model, "model", signature=signature)



# COMMAND ----------

latest_run = mlflow.search_runs(experiment_ids=run.info.experiment_id).iloc[0] # get the latest run
print(f"Testing f1 score is {latest_run['metrics.my_f1_testing']}")

# COMMAND ----------

# Catalog and schema to register models at
catalog = "edav_dev_csels"
schema = "towerscout_test_schema"

# COMMAND ----------

# get run id of latest run
run_id = run.info.run_id

model_name = "mflow_demo_model"

# register mode & get registered model metadata
model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model", # path to logged artifact folder called models
    name=f"{catalog}.{schema}.{model_name}"
    )

# COMMAND ----------

model_name = model_metadata.name

# set model alias to staging
client.set_registered_model_alias(
    name=model_name, 
    alias="Staging", 
    version=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train and tune a model
# MAGIC
# MAGIC Hyperopt is a Python optimzation library designed for optimizing functions with exotic parameter spaces in a distributed manner. For example, functions with real-valued and discrete-valued parameters as well as categorical parameters. This flexibility is needed for hyperparameter optimization. 

# COMMAND ----------

# define hyperparameter search space
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 140, 10)), 
  'max_depth':  scope.int(hp.quniform('max_depth', 2, 10, 2)),
  'random_state': 123, # Set seed for reproducablity 
}

def train_model_hyperopt(params):
  mlflow.sklearn.autolog() # use autologging
  
  with mlflow.start_run(nested=True):
    
    # Create and train model
    model = RandomForestClassifier(**params) # pass params from hopt
    model.fit(X_train, y_train)

    # Inference on test data
    y_pred = model.predict(X_test)

    # no need to infer signature or log model since we use autologging
    
    # log error metric
    f1 = f1_score(y_pred, y_test, average="weighted")
    mlflow.log_metric("my_f1_testing", f1)

    # Set the loss to -1*f1 so fmin maximizes the f1_score
    return {'status': STATUS_OK, 'loss': -1*f1}


# COMMAND ----------

# MAGIC %md
# MAGIC Hyperopt has direct spark integration so it allows you to paralellize the search algorithm being used via spark clusters. Greater parallelism results in quicker tuning, but a less optimal hyperparameter search since the search algorithm is iterative and each iteration is informed by the results of previous iterations. A reasonable value for parallelism is the square root of `max_evals` or 1/10 the value of `max_evals`.

# COMMAND ----------

spark_trials = SparkTrials(parallelism=4)

# COMMAND ----------

# MAGIC %md
# MAGIC Call `fmin` within a MLflow run context so that each hyperparameter configuration gets logged as a sub run of a parent run called "hyperopt_rf_models" in the experiments dashboard.

# COMMAND ----------


with mlflow.start_run(run_name='hyperopt_rf_models'):
  best_params = fmin(
    fn=train_model_hyperopt, # objective func
    space=search_space, # parameter space
    algo=tpe.suggest, # Tree of Parzen Estimators
    max_evals=16, # max obj func evals
    trials=spark_trials, # num paralell evaluations
  )

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.my_f1_testing DESC']).iloc[0] # sort by testing f1_socre we logged
print(f'Testing F1 score of best run: {best_run["metrics.my_f1_testing"]}')

# COMMAND ----------

run_id = best_run.run_id
model_name = "mflow_demo_model_prod" # production model name

best_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model", # path to logged artifact folder called models
    name=f"{catalog}.{schema}.{model_name}" # name for model in catalog
    )

# COMMAND ----------

best_model_name = best_model_metadata.name

client.set_registered_model_alias(
    name=best_model_name, 
    alias="Produciton", 
    version=1
)
