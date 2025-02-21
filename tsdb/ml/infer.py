"""
This module contains UDF's for inference for the TowerScout application
"""
from io import BytesIO
from json import loads
from typing import Any, Iterable, Protocol

import pandas as pd
from PIL import Image
from torch import no_grad, Tensor, cuda
from torch.nn import Module

import mlflow

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import DataFrame
import pyspark.sql.types as T

from tsdb.ml.datasets import ImageBinaryDataset
from tsdb.ml.utils import get_model_tags


class InferenceModelType(Protocol):  # pragma: no cover
    """
    A model class to wrap the model and provide the required methods to run
    distributed inference
    TODO: model instantiation logic should be moved to the model class
    """

    @property
    def model(self) -> Module:
        raise NotImplementedError

    @property
    def return_type(self) -> T.StructType:
        raise NotImplementedError

    def __call__(self, input) -> Tensor:  # dunder methods
        raise NotImplementedError

    def preprocess_input(self, input) -> Tensor:
        # torchvision.transforms
        raise NotImplementedError


YOLO_RETURN_TYPE = T.ArrayType(
            T.StructType([
                T.StructField("x1", T.FloatType(), True),
                T.StructField("y1", T.FloatType(), True),
                T.StructField("x2", T.FloatType(), True),
                T.StructField("y2", T.FloatType(), True),
                T.StructField("conf", T.FloatType(), True),
                T.StructField("class", T.IntegerType(), True),
                T.StructField("class_name", T.StringType(), True),
                T.StructField("secondary", T.FloatType(), True),
            ])
        )


MODEL_METADATA_STRUCT = T.StructType([
    T.StructField("yolo_model", T.StringType()),
    T.StructField("yolo_model_version", T.StringType()),
    T.StructField("efficientnet_model", T.StringType()),
    T.StructField("efficientnet_model_version", T.StringType())
])


TOWERSCOUT_IMAGE_METADATA_STRUCT = T.StructType([
        T.StructField("height", T.IntegerType()),
        T.StructField("width", T.IntegerType()),
        T.StructField("lat", T.DoubleType()),
        T.StructField("long", T.DoubleType()),
        T.StructField("image_id", T.IntegerType()),
        T.StructField("map_provider", T.StringType())
    ])


UDF_RETURN_TYPE = T.StructType([
        T.StructField("bboxes", YOLO_RETURN_TYPE),
        T.StructField("model_version", MODEL_METADATA_STRUCT),
        T.StructField("image_metadata", TOWERSCOUT_IMAGE_METADATA_STRUCT)
    ])


def make_towerscout_predict_udf(
    catalog: str,
    schema: str,
    yolo_alias: str = "aws",
    efficientnet_alias: str = "aws",
    batch_size: int = 100,
    num_workers: int = 2
) -> callable:
    """
    For a pandas UDF, we need the outer function to initialize the models
    plus any other objects we want to persist within the context of the UDF
    and the inner function to perform the inference process.
 
    Args:
        catalog: The catalog name.
        schema: The schema name.
        yolo_alias: The alias for the YOLO model in UC.
        efficientnet_alias: The alias for the EfficientNet model in UC.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of workers for the DataLoader.
 
    Returns:
        callable: A pandas UDF for inference.
    """

    # set unity catalog as registry to get models from
    mlflow.set_registry_uri("databricks-uc")
 
    yolo_model_name = f"{catalog}.{schema}.yolo_autoshape"
    en_model_name = f"{catalog}.{schema}.efficientnet"  
 
    # Retrieves models by alias
    yolo_detector = mlflow.pytorch.load_model(
            model_uri=f"models:/{yolo_model_name}@{yolo_alias}"
        )
 
    en_classifier = mlflow.pytorch.load_model(
            model_uri=f"models:/{en_model_name}@{efficientnet_alias}"
        )
    
    yolo_detector.eval()
    en_classifier.eval()

    if cuda.is_available():  # pragma: no cover
       en_classifier.cuda()
       yolo_detector.cuda()

    _, yolo_uc_version = get_model_tags(yolo_model_name, yolo_alias)
    _, en_uc_version = get_model_tags(en_model_name, efficientnet_alias)

    model_metadata = {
        "yolo_model": "yolo_autoshape",
        "yolo_model_version": yolo_uc_version,
        "efficientnet_model": "efficientnet",
        "efficientnet_model_version": en_uc_version
    }


    @pandas_udf(UDF_RETURN_TYPE)
    @no_grad()
    def predict_udf(image_bins: pd.Series) -> pd.Series:  # pragma: no cover
        """
        This predict_udf function is distributed across executors to perform inference.
 
        YOLOv5 library expects the following image formats: file, URI, numpy, PIL, OpenCV, torch tensor, multiple.
        - See `forward` method of Autoshape class in:
        https://github.com/ultralytics/yolov5/blob/master/models/common.py
        NOTE: Despite this do NOT pass tensors during inference (only training) 
        - See: https://github.com/ultralytics/yolov5/issues/7030#issuecomment-1078171092
       
        NOTE: No need to resize images for yolov5 lib as it does it for you
        - Source: letterbox and exif_transpose funcs in:
            https://github.com/ultralytics/yolov5/blob/master/models/common.py
 
        Args:
            image_bins: A partition of image binaries
 
        Returns: 
             Predicted labels and extracted image metadata.
        """
        pass
 
    return predict_udf
