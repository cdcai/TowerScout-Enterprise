# Databricks notebook source
# %pip install mosaicml-streaming
# %pip install optuna

# COMMAND ----------

from typing import Any
from collections import defaultdict

from uuid import uuid4

import torch
import torchvision

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import PIL

import numpy as np

from ultralytics.data.augment import Mosaic, Compose
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import xywh2xyxy
from streaming import StreamingDataset
from torch.utils.data import DataLoader

from tsdb.ml.utils import Hyperparameters
from tsdb.ml.data import YoloDataset, get_dataloader

# COMMAND ----------

hyperparams = Hyperparameters(
    lr0=0.1,
    momentum=0.9,
    weight_decay=0.1,
    batch_size=2,
    epochs=2,
    prob_H_flip=0.5,
    prob_V_flip=0.0,
    prob_mosaic=0.0,
)


cache_dir = "/local/cache/path/" + str(uuid4())
remote_dir = "/Volumes/edav_dev_csels/towerscout/data/mds_training_splits/test_image_gold/version=397/train"

dataloader = get_dataloader(cache_dir, remote_dir, hyperparams)

for i, image_batch in enumerate(dataloader):
    idx = image_batch["batch_idx"] == 0
    bboxes = xywh2xyxy(image_batch["bboxes"][idx]) * 640
    image = image_batch["img"][0]  #.permute(0,1,2)
    test_image = draw_bounding_boxes(image, bboxes, colors="red")
    #display(to_pil_image(image.permute(0,1,2)))
    display(to_pil_image(test_image))
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
