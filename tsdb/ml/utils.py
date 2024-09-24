from collections import namedtuple
from dataclasses import dataclass, asdict, field

from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql.types import Row
import pyspark.sql.functions as F

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