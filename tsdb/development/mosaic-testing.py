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

import PIL

import numpy as np

from ultralytics.data.augment import Mosaic, Compose
from ultralytics.utils.instance import Instances
from streaming import StreamingDataset
from torch.utils.data import DataLoader

from tsdb.ml.utils import Hyperparameters
from tsdb.ml.data import data_augmentation, ToTensor

# COMMAND ----------

class ModifiedMosaic(Mosaic):
    """
    A modified Mosaic augmentation object that inherets from the Mosaic class from Ultralytics.
    The sole modification is the removal of the 'buffer' parameter from the Mosaic class
    so that the 'buffer' is always the entire dataset.
    """
    def __init__(self, dataset, image_size, p, n):
        """
        NOTE: image_size determines the size of the images *comprising* the mosaic.
        So for an image size of DxD and for a mosaic of 4 images (2 x 2 grid of images)
        the output mosaic image will have a size of 2D x 2D.
        """
        super().__init__(dataset=dataset, imgsz=image_size, p=p, n=n)

    def get_indexes(self) -> list[int]:
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.
        This implementation removes the 'buffer' parameter and always uses the entire dataset

        This method selects random image indexes from the entire dataset.
        It is used to choose images for creating mosaic augmentations.

        Returns:
            (List[int]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        # select from any images in dataset
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]


class YoloDataset(StreamingDataset):
    """
    A dataset object that inherets from the StreamingDataset class with
    modifications to be compatible with the Ultralytics Mosaic 
    (not to be confused with MosaicML and its associated streaming library) augmentation object.
    See https://docs.mosaicml.com/projects/streaming/en/stable/how_to_guides/cifar10.html#Loading-the-Data
    for more details on working with custom MosaicML Streaming datasets.

    Args:
        remote: Remote path or directory to download the dataset from.
        local: str,
        shuffle: bool,
        hyperparameters: Hyperparameters,
        image_size: image size used to create the mosaic augmentation object
    """
    def __init__(
        self,
        remote: str,
        local: str,
        shuffle: bool,
        hyperparameters: Hyperparameters,
        image_size: int,
    ):
        super().__init__(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=hyperparameters.batch_size,
        )

        mosaic_aug = ModifiedMosaic(
            self, image_size=image_size, p=hyperparameters.prob_mosaic, n=4
        )

        flips = data_augmentation(
            prob_H_flip=hyperparameters.prob_H_flip,
            prob_V_flip=hyperparameters.prob_V_flip,
        )

        # compose Ultralytics transforms
        self.transforms = Compose([mosaic_aug] + flips + [ToTensor()])

    def __getitem__(self, index: int) -> Any:
        labels = self.get_image_and_label(index)
        return self.transforms(labels)

    def get_image_and_label(self, index: int) -> Any:
        """
        Get and return image & label information from the dataset.
        This function is added because the Ultralytics augmentation objects
        like Mosaic require the dataset object to have a method called
        get_image_and_label.
        (We add this to align with the protocol of the Ultralytics Mosiac object)
        """
        label = super().__getitem__(index)  # get row from dataset

        # move bboxes from "bboxes" key to "instances" key
        instances = Instances(
            bboxes=deepcopy(label.pop("bboxes")).reshape(-1, 4),
            # See: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py#L227
            # We set this value
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xyxy",
            normalized=True,
        )
        # from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py#L295
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation

        # rename "image_path" key to "im_file"
        label["im_file"] = label.pop("image_path")
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
    """
    result = defaultdict(list)

    for index, element in enumerate(data):
        element["instances"].convert_bbox(format="xyxy")
        bboxes = element.pop("instances")._bboxes.bboxes
        result["bboxes"].append(bboxes)

        for key, value in element.items():
            result[key].append(value)

        num_boxes = len(element["cls"])
        for _ in range(num_boxes):
            # yolo dataloader has this as a float not int
            result["batch_idx"].append(float(index))

    result["img"] = torch.stack(result["img"], dim=0)
    result["batch_idx"] = torch.tensor(result["batch_idx"])

    # Shape of resulting tensor should be (num bboxes in batch, 4)
    result["bboxes"] = torch.tensor(np.concatenate(result["bboxes"]))

    # Shape of resulting tensor should be (num bboxes in batch, 1)
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
    batch_size=1,
    epochs=2,
    prob_H_flip=1.0,
    prob_V_flip=1.0,
    prob_mosaic=1.0,
)


cache_dir = "/local/cache/path/" + str(uuid4())
remote_dir = "/Volumes/edav_dev_csels/towerscout/data/mds_training_splits/test_image_gold/version=377/train"

dataset = YoloDataset(
    local=cache_dir,
    remote=remote_dir,
    shuffle=True,
    hyperparameters=hyperparams,
    image_size=320,
)

dataloader = DataLoader(
    dataset=dataset, batch_size=hyperparams.batch_size, collate_fn=collate_fn_img
)

for i, batch in enumerate(dataloader):
    print(f"Image batch looks like: {batch['img'].shape}")
    print(f"Bboxes batch looks like: {batch['bboxes'].shape}")
    print(batch)
    to_pil_image = torchvision.transforms.ToPILImage()
    img = to_pil_image(batch["img"][0].transpose(0, 2))
    if i == 0:
        break

# COMMAND ----------

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


def denormalize_bounding_boxes(
    item: BoundingBox, width: int=640, height: int=640
) -> tuple[float, float, float, float]:
    return (
        item["x1"] * width,
        item["y1"] * height,
        item["x2"] * width,
        item["y2"] * height
    )


image_index = 0
selected_image = batch["img"][image_index]

image = batch["img"][image_index].to(torch.uint8).permute(2,0,1)
print(image.size())
bboxes = batch["bboxes"] 
print(bboxes)

test_image = draw_bounding_boxes(image, bboxes, colors="red")
display(to_pil_image(test_image))

# COMMAND ----------

cache_dir = "/local/cache/path/" + str(uuid4())

dataset = YoloDataset(
    local=cache_dir,
    remote=remote_dir,
    shuffle=False,
    hyperparameters=hyperparams,
    image_size=320,
)

dataloader = DataLoader(
    dataset=dataset, batch_size=1, collate_fn=collate_fn_img, shuffle=False
)

data_iter = iter(dataloader)
for i in range(1):
    batch = next(data_iter)

print(f"Image batch looks like: {batch}")

# COMMAND ----------

import matplotlib.pyplot as plt

data = dataset.get_item(2)
to_tens = ToTensor()
#data = to_tens(data)
#print(data["img"].dtype)
to_pil_image = torchvision.transforms.ToPILImage()
data["img"] = torch.tensor(np.ascontiguousarray(data["img"])).float() #/ 255
new_img = to_pil_image(data["img"].permute(2,0,1))
plt.imshow(new_img)
#display(new_img)  # for some reason output images are invereted in color? 

# COMMAND ----------

data = dataset.__getitem__(0)
#data = dataset.get_item(0)
#print(data)
print(data["instances"].convert_bbox(format="xyxy"))
print("Boxes", data["instances"]._bboxes.bboxes)

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
