import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow client
client = MlflowClient()

logged_model = f"https://davsynapseanalyticsdev.blob.core.windows.net/ddphss-csels/PD/TowerScout/Unstructured/model_params/en/b5_unweighted_best.pt"

run_name = "static_benchmark_model"

benchmark_model = mlflow.pyfunc.load_model(logged_model)

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(logged_model, "towerscout_baseline")
    run_id = run.info.run_id

    y_pred = model.predict(x_test)
    
    signature = mlflow.infer_signature(x_test, y_pred)

    mlflow.pyfunc.log_model(benchmark_model, run_name, signature=signature)


mlflow.set_experiment("/Workspace/Users/nzs0@cdc.gov/TowerScout_Experiment_Baseline")


run_id = "?"

ts_baseline_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model", # path to logged artifact folder called models

)


print(run_id)
