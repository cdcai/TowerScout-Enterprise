import mlflow
import torch
from torch.utils.data import DataLoader

from tsdb.ml.utils import UCModelName
from tsdb.ml.yolo import YoloModelTrainer


def model_promotion(
    challenger_uri: str,
    testing_dataloader: DataLoader,
    comparison_metric: str,
    uc_model_name: UCModelName,
    alias: str,
) -> None:  # pragma: no cover
    """
    Evaluates the model that has the specficied alias. Promotes the model with the
    specfied alias

    Args:
        challenger_uri: The URI for the challenger model. Found under the experiment it was logged under.
        testing_dataloader: The dataloader for the test dataset
        comparision_metric: The evaluation metric used to compare performance of the two models
        uc_model_name: The name to register the model under in Unity Catalog
        alias: The alias we are potentially promoting the model to
    Returns:
        None
    """
    mlflow.set_registry_uri("databricks-uc")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    challenger = mlflow.pytorch.load_model(challenger_uri)
    challenger.to(device)

    validator = YoloModelTrainer.get_validator(
        dataloader=testing_dataloader,
        training=False,
        device=device,
        args=challenger.args,
    )

    metrics = validator(challenger)
    # b/c we use Ultralytics valdiator the metrics dict has keys "metrics/{metric_name}(B)"
    challenger_test_metric = metrics[f"metrics/{comparison_metric}(B)"]

    # load current prod/champion model with matching alias
    champion = mlflow.pytorch.load_model(
        model_uri=f"models:/{str(uc_model_name)}@{alias}"
    )
    champion.to(device)

    metrics = validator(champion)
    champion_test_metric = metrics[f"metrics/{comparison_metric}(B)"]

    if challenger_test_metric > champion_test_metric:
        print(f"Promoting challenger model to {alias}.")

        # give alias to challenger model, alias is automatically removed from current champion model
        registration_info = mlflow.register_model(
            challenger_uri, name=str(uc_model_name)
        )
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=str(uc_model_name),
            alias=alias,
            version=registration_info.version,  # version of challenger model from when it was registered
        )
    else:
        print(
            f"Challenger model does not perform better than current {alias} model. Promotion aborted."
        )