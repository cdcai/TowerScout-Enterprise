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
import mlflow
from torch import nn
from PIL import Image
from tsdb.ml.efficientnet import EN_Classifier
from tsdb.ml.utils import get_model_tags, YOLOv5Detection
import pyspark.sql.types as T


class YOLOv5_Detector:
    def __init__(self, model: nn.Module, batch_size: int, uc_version: str):
        self.model = model
        self.batch_size = batch_size    
        self.uc_version = uc_version
        # follows the InferenceModelType protocol
        self.return_type = T.ArrayType(
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

    @classmethod
    def from_uc_registry(cls, model_name: str, alias: str, batch_size: int):
        """
        Create YOLOv5_Detector object using a registered model from UC Model Registry

        TODO: test
        """
        # IMPORTANT: when loading the model you must append the path to this directory to the system path so
        # Python looks there for the files/modules needed to load the yolov5 module
        model_tags, uc_version = get_model_tags(model_name, alias)
        catalog, schema, _ = model_name.split(".")

        try:
            yolo_version = model_tags["yolo_version"]
        except KeyError:
            print("YOLO version not found in model tags.")

        yolo_dep_path = f"/Volumes/{catalog}/{schema}/misc/yolo{yolo_version}"
        sys.path.append(yolo_dep_path)

        registered_model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}@{alias}"
        )

        return cls(registered_model, batch_size, uc_version)

    def predict(
        self,
        model_input: list[Image],
        secondary: EN_Classifier = None,
    ) -> list[list[YOLOv5Detection]]:
        results = []
        count = 0

        for i in range(0, len(model_input), self.batch_size):
            img_batch = model_input[i : i + self.batch_size]

            # retrain a copy of the images
            # TODO: remove this copying, we don't need to
            if secondary is not None:  # pragma: no cover
                img_batch2 = [img.copy() for img in img_batch]
            else:
                img_batch2 = [None] * len(img_batch)

            result_obj = self.model(img_batch)

            results_raw = result_obj.xyxyn

            for img, result in zip(img_batch2, results_raw):
                results_cpu = result.cpu().numpy().tolist()

                # secondary classifier processing
                if secondary is not None:  # pragma: no cover
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
