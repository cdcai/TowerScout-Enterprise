"""
This module contains helper functions that interact with MLFlow in Databricks.
"""
from dataclasses import dataclass, field
import mlflow


@dataclass
class MLFlowHelper:
    catalog: str
    schema: str
    registry_uri: str = "databricks-uc"
    registered_models: dict[str, str] = field(init=False) #[str, RegisteredModelSearch]

    def __post_init__(self) -> None:
        mlflow.set_registry_uri("databricks-uc")
        self.__initialize_models()

    def __initialize_models(self) -> None:
        """
        internal function to initialize the models dictionary
        """
        all_registered_models = mlflow.search_registered_models()

        project_registered_models = (
            model for model in all_registered_models
            if model.name.startswith(f"{self.catalog}.{self.schema}.")
        )
        self.registered_models = {
            model.name.split(".")[-1]: model for model in project_registered_models
        }
    
    def __getitem__(self, model_name: str):
        return self.registered_models[model_name]
