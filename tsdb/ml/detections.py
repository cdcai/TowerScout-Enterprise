#
# TowerScout
# A tool for identifying cooling towers from satellite and aerial imagery
#
# TowerScout Team:
# Karen Wong, Gunnar Mein, Thaddeus Segura, Jia Lu
#
# Licensed under CC-BY-NC-SA-4.0
# (see LICENSE.TXT in the root of the repository for details)
#

# YOLOv5 detector class

import sys
import os

import numpy as np
import mlflow
from torch import nn
from PIL import Image
from pyspark.sql import SparkSession

from tsdb.ml.efficientnet import ENClassifier
from tsdb.ml.utils import get_model_tags, YOLOv5Detection
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    IntegerType,
)


def initialize_yolov5_cluster(model_name, alias):
    model_tags = get_model_tags(model_name, alias)
    catalog, schema, _ = model_name.split(".")

    try:
        yolo_version = model_tags["yolo_version"]
    except KeyError:
        print("YOLO version not found in model tags.")

    # We need all nodes in the cluster to have access to this variable to load the model
    yolo_dep_path = f"dbfs:/Volumes/{catalog}/{schema}/misc/yolo{yolo_version}"

    return yolo_dep_path


class YOLOv5_Detector:
    def __init__(self, model: nn.Module, batch_size: int):
        self.model = model
        self.batch_size = batch_size

        # follows the InferenceModelType protocol
        self.return_type = StructType(
            [
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("conf", FloatType(), True),
                StructField("class", IntegerType(), True),
                StructField("class_name", StringType(), True),
                StructField("secondary", FloatType(), True),
            ]
        )

    @classmethod
    def from_uc_registry(cls, model_name: str, alias: str, batch_size: int):
        """
        Create YOLOv5_Detector object using a registered model from UC Model Registry
        """
        # IMPORTANT: when loading the model you must append the path to this directory to the system path so
        # Python looks there for the files/modules needed to load the yolov5 module

        model_tags = get_model_tags(model_name, alias)
        catalog, schema, _ = model_name.split(".")

        try:
            yolo_version = model_tags["yolo_version"]
        except KeyError:
            print("YOLO version not found in model tags.")

        # We need all nodes in the cluster to have access to this variable to load the model
        yolo_dep_path = f"dbfs:/Volumes/{catalog}/{schema}/misc/yolo{yolo_version}"
        sys.path.append(yolo_dep_path)

        # Retrieve registered model from mlflow now that path is set
        registered_model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}@{alias}"
        )

        return cls(registered_model, batch_size)

    def predict(
        self,
        model_input: list[Image],
        secondary: ENClassifier = None,
    ) -> list[list[YOLOv5Detection]]:
        results = []
        count = 0

        for i in range(0, len(model_input), self.batch_size):
            img_batch = model_input[i : i + self.batch_size]

            # retrain a copy of the images
            if secondary is not None:
                img_batch2 = [img.copy() for img in img_batch]
            else:
                img_batch2 = [None] * len(img_batch)

            result_obj = self.model(img_batch)

            results_raw = result_obj.xyxyn

            for img, result in zip(img_batch2, results_raw):
                results_cpu = result.cpu().numpy().tolist()

                # secondary classifier processing
                if secondary is not None:
                    # classifier will append its own prob to every detection
                    secondary.classify(img, results_cpu, batch_id=count)
                    count += 1

                tile_results = [
                    {
                        "x1": item[0],
                        "y1": item[1],
                        "x2": item[2],
                        "y2": item[3],
                        "conf": item[4],
                        "class": int(item[5]),
                        "class_name": result_obj.names[int(item[5])],
                        "secondary": item[6] if len(item) > 6 else 1,
                    }
                    for item in results_cpu
                ]
                results.append(tile_results)

        return results