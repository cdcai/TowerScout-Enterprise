from collections import namedtuple
from dataclasses import dataclass, asdict, field

from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql.types import Row
import pyspark.sql.functions as F

import logging
from logging.handlers import RotatingFileHandler
from logging import Logger

from pathlib import Path

from enum import Enum

from torch import nn

from petastorm.spark.spark_dataset_converter import SparkDatasetConverter

from mlflow import MlflowClient



SchemaInfo = namedtuple("SchemaInfo", ["name", "location"])

# using a dataclass instead results in sparkcontext error
FminArgs = namedtuple("FminArgs", ["fn", "space", "algo", "max_evals", "trials"])


class ValidMetric(Enum):
    """
    An Enum which is used to represent valid evaluation metrics for the model
    """

    BCE = nn.BCEWithLogitsLoss()
    MSE = nn.MSELoss()


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


def setup_logger(log_path: str, logger_name: str) -> tuple[Logger, RotatingFileHandler]:
    """
    Creates and returns a Logger object

    Args:
        log_path: Path to store the log file
    Returns:
        The Logger object and the RotatingFileHandler object
    """
    # TODO: Log file may become too large, may need to be partitioned by date/week/month

    # Create the log directory if it doesn't exist
    log_dir = str(Path(log_path).parent)

    # Set up logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    try:
        # Create a rotating file handler
        handler = RotatingFileHandler(log_path, maxBytes=1000000, backupCount=1)
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger and then you can use the logger
        logger.addHandler(handler)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise e

    return logger, handler


def cast_to_column(column: "ColumnOrName") -> Column:
    """
    Returns a column data type. Used so functions can flexibly accept
    column or string names.
    """
    if isinstance(column, str):
        column = F.col(column)

    return column