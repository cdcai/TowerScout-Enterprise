"""
This module contains UDF's for inference for the TowerScout application
"""

from typing import Any, Protocol, Union
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image
from torch import no_grad, Tensor, cuda, sigmoid
from torch.utils.data import DataLoader
from torch.nn import Module
from torchvision import transforms

import mlflow

import pyspark.sql.types as T
import pyspark.sql.functions as F
from models.common import Detections  # Detection object for YOLOv5 model

from tsdb.ml.datasets import ImageBinaryDataset
from tsdb.ml.utils import get_model_tags, cut_square_detection
from tsdb.ml.types import ImageMetadata


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
    T.StructType(
        [
            T.StructField("x1", T.FloatType(), True),
            T.StructField("y1", T.FloatType(), True),
            T.StructField("x2", T.FloatType(), True),
            T.StructField("y2", T.FloatType(), True),
            T.StructField("conf", T.FloatType(), True),
            T.StructField("class", T.IntegerType(), True),
            T.StructField("class_name", T.StringType(), True),
            T.StructField("secondary", T.FloatType(), True),
        ]
    )
)


MODEL_METADATA_STRUCT = T.StructType(
    [
        T.StructField("yolo_model", T.StringType()),
        T.StructField("yolo_model_version", T.StringType()),
        T.StructField("efficientnet_model", T.StringType()),
        T.StructField("efficientnet_model_version", T.StringType()),
    ]
)


TOWERSCOUT_IMAGE_METADATA_STRUCT = T.StructType(
    [
        T.StructField("lat", T.DoubleType()),
        T.StructField("long", T.DoubleType()),
        T.StructField("width", T.IntegerType()),
         T.StructField("height", T.IntegerType())
    ]
)


UDF_RETURN_TYPE = T.StructType(
    [
        T.StructField("bboxes", YOLO_RETURN_TYPE),
        T.StructField("model_version", MODEL_METADATA_STRUCT),
        T.StructField("image_metadata", TOWERSCOUT_IMAGE_METADATA_STRUCT),
        T.StructField("image_id", T.IntegerType()),
        T.StructField("map_provider", T.StringType()),
    ]
)


def inference_collate_fn(
    data: list[dict[str, ImageMetadata]]
) -> dict[str, Union[Image, dict[str, Any]]]:
    """
    A collate function for DataLoaders used with the ImageBinaryDataset object.
    Collates the data into a batch for inference UDF.

    Args:
        data: The ImageMetadata objects to collate into a batch

    Retruns:
        A dictionary with the following keys:
        - images: A list of PIL images
        - images_metadata: A list of ImageMetadata objects with the PIL image object removed
    """
    batch = defaultdict(list)
    for item in data:
        batch["images"].append(item.pop("image"))
        batch["images_metadata"].append(item)

    return batch


def apply_secondary_model(
    secondary_model: Module,
    image: Image,
    detections: list[np.array],
    min_conf: float = 0.25,
    max_conf: float = 0.65,
) -> None:
    """
    A function to apply the secondary model to the detections from the YOLO model. The function
    first crops the image based on the bounding box predicted by the YOLO model, then applies
    the secondary model to the cropped image to determine the probablity the image contains a cooling tower
    and appends the computed probability to the detection array.

    Args:
        secondary_model: the secondary model to apply to the cropped image
        image: the image to crop
        detections: list of the detections from the YOLO model for the input image
        min_conf: the minimum confidence to apply the secondary model
        max_conf: the maximum confidence to apply the secondary model
    """
    transform = transforms.Compose(
        [
            transforms.Resize([456, 456]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5553, 0.5080, 0.4960), std=(0.1844, 0.1982, 0.2017)
            ),
        ]
    )

    for detection in detections:
        x1, y1, x2, y2, conf = detection[0:5]

        # Use secondary model only for certain confidence range
        if conf >= min_conf and conf <= max_conf:
            bbox_cropped_image = cut_square_detection(image, x1, y1, x2, y2)

            # apply transformations
            input = transform(bbox_cropped_image).unsqueeze(0)

            if cuda.is_available():  # pragma: no cover
                input = input.cuda()

            # subtract from 1 because the secondary has class 0 as tower
            output = 1 - sigmoid(secondary_model(input).cpu()).item()
            p2 = output
        elif conf < min_conf:
            # set secondary classifier probability to 0
            p2 = 0
        else:
            # if >= max_conf set secondary classifier probability to 1
            p2 = 1

        detection.append(p2)

    return


def parse_yolo_detections(
    images: list[Image],
    yolo_results: Detections,
    secondary_model: Module = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    A function to parse the detections from the YOLO model by converting them into a list
    of dicts with the following keys:
    - x1: the x1 coordinate of the bounding box
    - y1: the y1 coordinate of the bounding box
    - x2: the x2 coordinate of the bounding box
    - y2: the y2 coordinate of the bounding box
    - conf: the YOLO model confidence of the detection
    - class: the class of the detection
    - class_name: the name of the class of the detection
    - secondary: the secondary model confidence of the detection (if a secondary model is supplied)

    Args:
        images: the list of PIL images
        yolo_results: the Detections object from the YOLO model
        secondary_model: the secondary model used to evaluate the detections
        **kwargs: additional keyword arguments to pass to the secondary model
    Returns:
        A list of dicts with the keys from above.
    """
    parsed_results = []
    batch_detections = yolo_results.xyxyn

    for image, image_detections in zip(images, batch_detections):
        image_detections = image_detections.cpu().numpy().tolist()

        if secondary_model is not None:
            apply_secondary_model(secondary_model, image, image_detections, **kwargs)

        image_results = [
            {
                "x1": item[0],
                "y1": item[1],
                "x2": item[2],
                "y2": item[3],
                "conf": item[4],
                "class": int(item[5]),
                "class_name": yolo_results.names[int(item[5])],
                "secondary": item[6] if len(item) > 6 else 1,
            }
            for item in image_detections
        ]

        parsed_results.append(image_results)

    return parsed_results


def make_towerscout_predict_udf(
    catalog: str,
    schema: str,
    yolo_alias: str = "aws",
    efficientnet_alias: str = "aws",
    batch_size: int = 100,
    num_workers: int = 2,
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
        "efficientnet_model_version": en_uc_version,
    }

    @F.pandas_udf(UDF_RETURN_TYPE)
    @no_grad()
    def predict_udf(image_bins: pd.Series) -> pd.DataFrame:
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
        bin_dataset = ImageBinaryDataset(image_bins)
        loader = DataLoader(
            bin_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=inference_collate_fn,
        )

        outputs = []

        for batch in loader:
            yolo_output = yolo_detector(batch["images"])
            parsed_results = parse_yolo_detections(
                batch["images"], yolo_output, en_classifier
            )

            for parsed_result, image_metadata in zip(parsed_results, batch["images_metadata"]):
                map_provider = image_metadata.pop("map_provider")
                image_id = image_metadata.pop("image_id")
                
                outputs.append(
                    {
                        "bboxes": parsed_result,
                        "model_version": model_metadata,
                        "image_metadata": image_metadata,
                        "image_id": image_id,
                        "map_provider": map_provider,
                    }
                )

        return pd.DataFrame(outputs)

    return predict_udf


def inference_collate_fn(data: list[dict[str, ImageMetadata]]) -> dict[str, Union[Image, dict[str, Any]]]:
    """
    A collate function for DataLoaders used with the ImageBinaryDataset object. 
    Collates the data into a batch for inference UDF.

    Args:
        data: The ImageMetadata objects to collate into a batch
    
    Retruns:
        A dictionary with the following keys:
        - images: A list of PIL images
        - images_metadata: A list of ImageMetadata objects with the PIL image object removed
    """
    batch = defaultdict(list)
    for item in data:
        batch["images"].append(item.pop("image"))
        batch["images_metadata"].append(item)

    return batch


def apply_secondary_model(
    secondary_model: Module,
    image: Image,
    detections: list[np.array],
    min_conf: float = 0.25,
    max_conf: float = 0.65,
) -> None:
    """
    A function to apply the secondary model to the detections from the YOLO model. The function 
    first crops the image based on the bounding box predicted by the YOLO model, then applies 
    the secondary model to the cropped image to determine the probablity the image contains a cooling tower
    and appends the computed probability to the detection array.

    Args:
        secondary_model: the secondary model to apply to the cropped image
        image: the image to crop
        detections: list of the detections from the YOLO model for the input image
        min_conf: the minimum confidence to apply the secondary model
        max_conf: the maximum confidence to apply the secondary model
    """
    transform = transforms.Compose(
        [
            transforms.Resize([456, 456]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5553, 0.5080, 0.4960), std=(0.1844, 0.1982, 0.2017)
            ),
        ]
    )

    for detection in detections:
        x1, y1, x2, y2, conf = detection[0:5]

        # Use secondary model only for certain confidence range
        if conf >= min_conf and conf <= max_conf:
            bbox_cropped_image = cut_square_detection(image, x1, y1, x2, y2)

            # apply transformations
            input = transform(bbox_cropped_image).unsqueeze(0)

            if cuda.is_available():  # pragma: no cover
                input = input.cuda()

            # subtract from 1 because the secondary has class 0 as tower
            output = 1 - sigmoid(secondary_model(input).cpu()).item()
            p2 = output
        elif conf < min_conf:
            # set secondary classifier probability to 0
            p2 = 0
        else:
            # if >= max_conf set secondary classifier probability to 1
            p2 = 1

        detection.append(p2)
    
    return


def parse_yolo_detections(
    images: list[Image],
    yolo_results: Detections,
    secondary_model: Module = None,
    **kwargs
) -> list[dict[str, Any]]:
    """
    A function to parse the detections from the YOLO model by converting them into a list
    of dicts with the following keys:
    - x1: the x1 coordinate of the bounding box
    - y1: the y1 coordinate of the bounding box
    - x2: the x2 coordinate of the bounding box
    - y2: the y2 coordinate of the bounding box
    - conf: the YOLO model confidence of the detection
    - class: the class of the detection
    - class_name: the name of the class of the detection
    - secondary: the secondary model confidence of the detection (if a secondary model is supplied)

    Args:
        images: the list of PIL images
        yolo_results: the Detections object from the YOLO model
        secondary_model: the secondary model used to evaluate the detections
        **kwargs: additional keyword arguments to pass to the secondary model
    Returns:
        A list of dicts with the keys from above.
    """
    parsed_results = []
    batch_detections = yolo_results.xyxyn

    for image, image_detections in zip(images, batch_detections):
        image_detections = image_detections.cpu().numpy().tolist()
        
        if secondary_model is not None:
            apply_secondary_model(secondary_model, image, image_detections, **kwargs)

        image_results = [
                    {
                        "x1": item[0],
                        "y1": item[1],
                        "x2": item[2],
                        "y2": item[3],
                        "conf": item[4],
                        "class": int(item[5]),
                        "class_name": yolo_results.names[int(item[5])],
                        "secondary": item[6] if len(item) > 6 else 1,
                    }
                    for item in image_detections
                ]

        parsed_results.append(image_results)

    return parsed_results
