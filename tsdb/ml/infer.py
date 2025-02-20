"""
This module contains UDF's for inference for the TowerScout application
"""
from collections import defaultdict
from typing import Any, Protocol, Union

import pandas as pd
import numpy as np
from PIL import Image

from torch import no_grad, Tensor, cuda, sigmoid
from torch.nn import Module
from torchvision import transforms

import pyspark.sql.types as T
from models.common import Detections  # Detection object for YOLOv5 model

from tsdb.ml.utils import cut_square_detection
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


MODEL_VERSION_STRUCT = T.StructType([
    T.StructField("yolo_model", T.StringType()),
    T.StructField("yolo_model_version", T.StringType()),
    T.StructField("efficientnet_model", T.StringType()),
    T.StructField("efficientnet_model_version", T.StringType())
])


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

