from collections import namedtuple
from dataclasses import dataclass, field
from typing import TypedDict

from enum import Enum

from torch import nn

from petastorm.spark.spark_dataset_converter import SparkDatasetConverter

from mlflow import MlflowClient

from logging import Logger

class ValidMetric(Enum):
    """
    An Enum which is used to represent valid evaluation metrics for the model
    """

    BCE = nn.BCEWithLogitsLoss()
    MSE = nn.MSELoss()


# using a dataclass instead results in sparkcontext error
FminArgs = namedtuple("FminArgs", ["fn", "space", "algo", "max_evals", "trials"])

@dataclass
class SplitConverters:
    """
    A class to hold the spark dataset converters for the training, testing
    and validation sets

    Attributes:
        train: The spark dataset converter for the training dataset
        val: The spark dataset converter for the validation dataset
        test: The spark dataset converter for the testing dataset
    """

    train: SparkDatasetConverter = None
    val: SparkDatasetConverter = None
    test: SparkDatasetConverter = None


@dataclass
class TrainingArgs:
    """
    A class to represent model training arguements

    Attributes:
        objective_metric:The evaluation metric we want to optimize
        epochs: Number of epochs to optimize model over
        batch_size: The size of the minibatchs passed to the model
        report_interval: Interval to log metrics during training
        metrics: Various model evaluation metrics we want to track
    """

    objective_metric: str = "recall"  # will be selected option for the drop down
    epochs: int = 2
    batch_size: int = 4
    report_interval: int = 5
    metrics: list[ValidMetric] = field(default_factory=dict)


@dataclass
class PromotionArgs:
    """
    A class to represent model promotion arguements

    Attributes:
        objective_metric: The evaluation metric we want to optimize
        batch_size: The size of the minibatchs passed to the model
        metrics: Various model evaluation metrics we want to track
        model_version: The version of the model that is the challenger
        model_name: The name of the model
        challenger_metric_value: The value of the objective metric achieved by the challenger model on the test dataset
        alias: The alias we are promoting the model to
        test_conv: The converter for the test dataset
    """

    objective_metric: str = "recall"
    batch_size: int = 4
    metrics: list[ValidMetric] = field(default_factory=list)
    model_version: int = 1
    model_name: str = "ts"
    challenger_metric_value: float = 0
    alias: str = "staging"
    test_conv: SparkDatasetConverter = None
    client: MlflowClient = None
    logger: Logger = None


@dataclass
class OptimizerArgs:
    """
    A class to represent YOLO model optimizer arguements

    Attributes:
        optimizer_name: The name of optimzer to use SGD, Adam, etc
        lr0: The initial learning rate
        momentum: Momentum parameter
    """
    optimizer_name: str = "auto"
    lr0: float = 0.001
    momentum: float = 0.9

class YOLOv5Detection(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_: int
    class_name: str
    secondary: int

def cut_square_detection(img, x1, y1, x2, y2):
    w,h = img.size

    # first, convert detection fractional coordinates into pixels
    x1 *= w
    x2 *= w
    y1 *= h
    y2 *= h

    # compute width and height of cut area
    wc = x2-x1
    hc = y2-y1
    size = int(max(wc,hc)*1.5+(25*640/w)) # 25 is adjusted by image size (Google is 1280, Bing 640)

    # now square it
    cy = (y1+y2)/2.0
    y1 = cy - size/2.0
    y2 = cy + size/2.0
    
    cx = (x1+x2)/2.0
    x1 = cx - size/2.0
    x2 = cx + size/2.0

    # clip to picture
    x1 = max(0,x1)
    x2 = min(w,x2)
    y1 = max(0,y1)
    y2 = min(h,y2)

    return img.crop((x1, y1, x2, y2))

def get_model_tags(model_name: str, alias: str) -> tuple[dict[str, str], str]:
    """
    Returns the tags for the model with the given model name and alias
    along with the model version.
    Note we do not unit test this function as `get_model_version_by_alias`
    and `get_model_version` are unit tested in the MLflow git repo already at
    https://github.com/mlflow/mlflow/blob/8c07dc0f604565bec29358526db461ca4f842bb5/tests/tracking/test_client.py#L1532
    and 
    https://github.com/mlflow/mlflow/blob/8c07dc0f604565bec29358526db461ca4f842bb5/tests/store/model_registry/test_rest_store.py#L306
    respectively.
    """
    client = MlflowClient()
    model_version_info = client.get_model_version_by_alias(
        name=model_name, alias=alias
    )
    model_version = model_version_info.version
    model_version_details = client.get_model_version(
        name=model_name, version=model_version
    )
    model_tags = model_version_details.tags

    return model_tags, model_version
