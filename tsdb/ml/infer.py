from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from typing import Any, Iterable

from torch.utils.data import DataLoader

import pandas as pd

from tsdb.ml.models import InferenceModelType
from tsdb.ml.data_processing import TowerScoutDataset

def ts_model_udf(model_fn: InferenceModelType, batch_size: int, return_type: StructType) -> DataFrame:
    """
    A pandas UDF for distributed inference with a PyTorch model.

    Args:
        model_fn (InferenceModelType): The PyTorch model.
        batch_size (int): Batch size for the DataLoader.
        return_type (StructType): Return type for the UDF.

    Returns:
        DataFrame: DataFrame with predictions.
    """
    @torch.no_grad()
    def predict(content_series_iter: Iterable[Any]):
        """
        Predict function to be used within the pandas UDF.

        Args:
            content_series_iter: Iterator over content series.

        Yields:
            DataFrame: DataFrame with predicted labels.
        """
        model = model_fn()  # Load the model
        for content_series in content_series_iter:
            # Create dataset object to apply transformations
            dataset: Dataset = TowerScoutDataset(list(content_series))
            # Create PyTorch DataLoader
            loader = DataLoader(dataset, batch_size=batch_size)
            for image_batch in loader:
                # Perform inference on batch
                output = model(image_batch)
                predicted_labels = output.tolist()
                yield pd.Series(predicted_labels)

    return pandas_udf(return_type, PandasUDFType.SCALAR_ITER)(predict)