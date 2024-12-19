# Databricks notebook source
# MAGIC %pip install mosaicml-streaming

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
    .limit(1000)
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
