# Databricks notebook source
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
import numpy as np
import mlflow
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.entities.model_registry import ModelVersion
from torch import nn
from typing import Self
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    IntegerType,
)


class YOLOv5_Detector(PythonModel):
    def __init__(self, model: nn.Module, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.client = MlflowClient()

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
    def from_uc_registry(cls, model_name: str, alias: str) -> Self:
        # IMPORTANT: when loading the model you must append the path to this directory to the system path so
        # Python looks there for the files/modules needed to load the yolov5 module
        client = MlflowClient()
        model_version_info = client.get_model_version_by_alias(
            name=model_name, alias=alias
        )
        model_version = model_version_info.version
        model_version_details = client.get_model_version(
            name=model_name, version=model_version
        )
        model_tags = model_version_details.tags

        try:
            yolo_version = model_tags["yolo_version"]
        except KeyError:
            print("YOLO version not found in model tags.")

        yolo_dep_path = f"/Volumes/edav_dev_csels/towerscout_test_schema/ultralytics_yolo{yolo_version}_master"
        sys.path.append(yolo_dep_path)

        registered_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}@{alias}"
        )

        return registered_model

    def register_model(
        self, model_name: str, run_id: str, artifact_path: str
    ) -> ModelVersion:
        registered_model_metadata = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}",  # path to logged artifact folder called models
            name=model_name,
        )

        # set YOLO model version tag
        self.client.set_model_version_tag(
            name=model_name,
            version=registered_model_metadata.version,
            key="yolo_version",
            value="v5",
        )

        return registered_model_metadata

    def set_model_alias(self, model_name: str, alias: str, model_version: str) -> None:
        self.client.set_registered_model_alias(
            name=model_name, alias=alias, version=model_version
        )

    def preprocess_input(self, model_input: np.ndarray[np.ndarray]) -> list[np.ndarray]:
        # the model expects a list of images: list of np arrays or Image objects.
        # see: ultralytics_yolov5_master/models/common.py
        return [model_input[j] for j in range(len(model_input))]

    def predict(
        self,
        context: PythonModelContext,
        model_input: np.ndarray[np.ndarray],
        secondary: nn.Module = None,
    ) -> list[dict[str, float]]:
        results = []
        count = 0
        model_input = self.preprocess_input(model_input)

        for i in range(0, len(model_input), self.batch_size):
            img_batch = model_input[i : i + self.batch_size]

            # retain a copy of the images
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

                # record the detections in the tile
                # boxes = []
                # for tr in tile_results:
                #     box = "0 " + \
                #         str((tr['x1']+tr['x2'])/2) + \
                #         " "+str((tr['y1']+tr['y2'])/2) + \
                #         " "+str(tr['x2']-tr['x1']) +\
                #         " "+str(tr['y2']-tr['y1'])+"\n"
                #     boxes.append(box)
                # tile['detections'] = boxes

        return results
