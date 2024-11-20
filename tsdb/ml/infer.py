"""
This module contains UDF's for inference for the TowerScout application
"""
from io import BytesIO
from json import loads
from typing import Any, Iterable

import pandas as pd
from PIL import Image
from torch import no_grad

from mlflow import set_registry_uri

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import DataFrame
import pyspark.sql.types as T

from tsdb.ml.models import InferenceModelType
from tsdb.ml.detections import YOLOv5_Detector
from tsdb.ml.efficientnet import ENClassifier


def make_towerscout_predict_udf(
    catalog: str,
    schema: str,
    yolo_alias: str = "aws",
    efficientnet_alias: str = "aws",
    batch_size: int = 100
) -> DataFrame:
    """
    For a pandas UDF, we need the outer function to initialize the models
    and the inner function to perform the inference. Process. For more
    information, see the following reference by NVIDIA:
    - 

    Args:
        model_fn (InferenceModelType): The PyTorch model.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataFrame: DataFrame with predictions.
    """ 
    set_registry_uri("databricks-uc")

    yolo_model_name = f"{catalog}.{schema}.yolo_autoshape" 
    en_model_name = f"{catalog}.{schema}.efficientnet"  

    # Retrieves models by alias and create inference objects
    yolo_detector = YOLOv5_Detector.from_uc_registry(
        model_name=yolo_model_name,
        alias=yolo_alias,
        batch_size=batch_size,
    )

    # We nearly always use efficientnet for classification but you don't have to
    en_classifier = ENClassifier.from_uc_registry(
        model_name=en_model_name,
        alias=efficientnet_alias
    )

    @no_grad()
    def predict(content_series_iter: Iterable[Any]):
        """
        This predict function is distributed across executors to perform inference.

        YOLOv5 library expects the following image formats:
        For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images
        - Source: https://github.com/ultralytics/yolov5/blob/master/models/common.py
        
        The ultralytics lib accepts the following image formats:
        - Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py 
        

        # No need to resize for yolov5 lib as it does it for you 
        - Source: letterbox and exif_transpose funcs in:
            https://github.com/ultralytics/yolov5/blob/master/models/common.py

        Args:
            content_series_iter: Iterator over content series.

        Yields:
            DataFrame: DataFrame with predicted labels.
        """
        for content_series in content_series_iter:
            # Create dataset object to apply transformations
            image_batch = [
                Image.open(BytesIO(content)).convert("RGB")
                for content in content_series
            ]

            # Perform inference on batch
            output = yolo_detector.predict(model_input=image_batch, secondary=en_classifier)
            yield pd.Series(output)

    return pandas_udf(yolo_detector.return_type, PandasUDFType.SCALAR_ITER)(predict)
