# Databricks notebook source
# MAGIC %pip install mosaicml-streaming
# MAGIC %pip install optuna

# COMMAND ----------

from typing import Any
from collections import defaultdict
import random
from copy import deepcopy

from uuid import uuid4

import torch
import torchvision

import numpy as np
import cv2

from ultralytics.data.augment import Mosaic, Compose
from ultralytics.utils.instance import Instances
from streaming import StreamingDataset
from torch.utils.data import DataLoader

from tsdb.ml.utils import Hyperparameters
from tsdb.ml.data import data_augmentation, ToTensor

# COMMAND ----------

class TowerScoutMosaic(Mosaic):
    def __init__(self, dataset, imgsz, p, n):
        super().__init__(dataset=dataset, imgsz=imgsz, p=p, n=n)

    def get_indexes(self):
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.
        This implementation removes the 'buffer' parameter and always uses the entire dataset

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.

        Returns:
            (List[int]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        # select any images
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]


class YoloDataset(StreamingDataset):
    def __init__(
        self,
        remote: str,
        local: str,
        shuffle: bool,
        hyperparameters: Hyperparameters,
        image_size: int = 640,
    ):
        super().__init__(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=hyperparameters.batch_size,
        )
        mosaic_aug = TowerScoutMosaic(
            self, imgsz=image_size, p=hyperparameters.prob_mosaic, n=4
        )
        flips = data_augmentation(
            prob_H_flip=hyperparameters.prob_H_flip,
            prob_V_flip=hyperparameters.prob_V_flip,
        )
        #self.transforms = Compose([mosaic_aug] + flips + [ToTensor()])
        self.transforms = Compose([mosaic_aug] + flips + [ToTensor()])

    def __getitem__(self, index: int) -> Any:
        labels = self.get_image_and_label(index)
        labels = self.transforms(labels)
        return labels

    def get_image_and_label(self, index: int) -> Any:
        """Get and return image & label information from the dataset."""
        label = super().__getitem__(index)  # get row from dataset
        instances = Instances(
            bboxes=deepcopy(label["bboxes"]).reshape(-1, 4),
            segments=np.zeros((0, 1000, 2), dtype=np.float32),  # dummy arg, remove later. seems like overloading will be needed
            bbox_format="xyxy",
            normalized=True,
        )

        # from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py#L295
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation

        label["im_file"] = label["image_path"]
        label["instances"] = instances
        return label


def collate_fn_img(data: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Function for collating data into batches. Some additional
    processing of the data is done in this function as well
    to get the batch into the format expected by the Ultralytics
    DetectionModel class.

    Args:
        data: The data to be collated a batch
        transforms: Ultralytics transforms applied to the np array images
    Returns: A dictionary containing the collated data in the formated
            expected by the Ultralytics DetectionModel class

    TODO: DELETE HERE
    """
    result = defaultdict(list)

    for index, element in enumerate(data):
        bboxes = element["instances"]._bboxes.bboxes
        result["bboxes"].append(bboxes)
        for key, value in element.items():
            result[key].append(value)

        num_boxes = len(element["cls"])

        for _ in range(num_boxes):
            result["batch_idx"].append(
                float(index)
            )  # yolo dataloader has this as a float not int

    result["img"] = torch.stack(result["img"], dim=0)
    result["batch_idx"] = torch.tensor(result["batch_idx"])

    # Shape of resulting tensor should be (num_bboxes_in_batch, 4)
    result["bboxes"] = torch.tensor(np.concatenate(result["bboxes"]))

    # Shape of resulting tensor should be (num_bboxes_in_batch, 1)
    # Call np.concatenate to avoid calling tensor on a list of np arrays but instead just one 2d np array
    result["cls"] = torch.tensor(np.concatenate(result["cls"])).reshape(-1, 1)

    result["im_file"] = tuple(result["im_file"])
    result["ori_shape"] = tuple(result["ori_shape"])
    result["resized_shape"] = tuple(result["resized_shape"])
    result["ratio_pad"] = tuple(result["ratio_pad"])

    return dict(result)


hyperparams = Hyperparameters(
    lr0=0.1,
    momentum=0.9,
    weight_decay=0.1,
    batch_size=3,
    epochs=2,
    prob_H_flip=0.9,
    prob_V_flip=0.9,
    prob_mosaic=.0,
)


cache_dir = "/local/cache/path/" + str(uuid4())
remote_dir = "/Volumes/edav_dev_csels/towerscout/data/mds_training_splits/test_image_gold/version=377/train"
dataset = YoloDataset(
    local=cache_dir, remote=remote_dir, shuffle=True, hyperparameters=hyperparams, image_size=640
)

dataloader = DataLoader(dataset=dataset, batch_size=hyperparams.batch_size, collate_fn=collate_fn_img)

for i, batch in enumerate(dataloader):
    print(f"Image batch looks like: {batch['img'].shape}")
    print(f"Bboxes batch looks like: {batch['bboxes'].shape}")
    to_pil_image = torchvision.transforms.ToPILImage()
    img = to_pil_image(batch["img"][0].transpose(0, 2))
    if i == 1:
        break

# COMMAND ----------

import PIL
display(PIL.ImageOps.invert(img))  # for some reason output images are invereted in color?

# COMMAND ----------

import numpy as np
from shutil import rmtree
from uuid import uuid4
import pyspark.sql.functions as F
from pyspark.sql.column import Column

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel
from tsdb.ml.utils import OptimizerArgs
from tsdb.ml.yolo_trainer import YoloModelTrainer
from tsdb.ml.utils import OptimizerArgs
from tsdb.preprocessing.preprocess import get_dataloader, collate_fn_img, convert_to_mds

# COMMAND ----------

def get_proper_path(path: "ColumnOrName") -> Column:
    # remove the dbfs: prefix as that causes errors when reading image with PIL
    file_with_extension = F.element_at(F.split(path, ":"), 2)
    return file_with_extension

df = (
    spark
    .read
    .format("delta")
    .table("edav_dev_csels.towerscout.test_image_silver")
    .selectExpr("image_path as im_file", "bboxes")
    .where("processing_time > '2024-12-18'")  # to ignore rows whose images have been deleted from datalake due to pipeline errors
    .limit(10)
)


df = df.withColumn("im_file", get_proper_path("im_file"))
df = df.filter(F.size(F.col("bboxes")) > 0)  # filter out rows with no bboxes detected
df = df.withColumn("cls", F.expr("transform(bboxes, x -> float(0))") )  # for each bbox create a label `0`
df = df.withColumn("bboxes", F.expr("transform(bboxes, x -> array(x.x1, x.y1, x.x2, x.y2))") )  # only keep box coord keys from bboxes struct
df = df.withColumn("bboxes", F.flatten(F.col("bboxes")) )
display(df)

# COMMAND ----------

!rm -rf /tmp/ztm8/mosaicml/

# COMMAND ----------

out_root = '/tmp/ztm8/mosaicml/'

columns = {
    'im_file': 'str',
    'img': 'jpeg',
    'bboxes': 'ndarray:float32',
    'cls': 'ndarray:float32',
    'ori_shape': 'ndarray:uint32'
}

compression = 'zstd'

convert_to_mds(df, columns, compression, out_root)

# COMMAND ----------

remote_dir = out_root

local_dir = '/local/cache/path2'

batch_size = 32
dataloader = get_dataloader(local_dir, remote_dir, batch_size)

# COMMAND ----------

for i, batch in enumerate(dataloader):
    print(batch)
    break

# COMMAND ----------

model_yaml = "yolov8n.yaml"
model_pt = "yolov8n.pt"

args = get_cfg()  # used to get hyperparams for model and other stuff from some config file
model = DetectionModel(cfg=model_yaml, verbose=False)
weights, _ = attempt_load_one_weight(model_pt)
model.load(weights)
model.nc = 1
model.names ='ct'
model.args = args
model.args.single_cls = True

optimizer_args = OptimizerArgs(optimizer_name="Adam", lr0=0.001, momentum=0.99)
yolo_trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=model)

# COMMAND ----------

for i, batch in enumerate(dataloader):
    metrics = yolo_trainer.training_step(batch)
    print(f"Training metrics: {metrics}")
