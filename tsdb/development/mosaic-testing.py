# Databricks notebook source
# MAGIC %pip install mosaicml-streaming

# COMMAND ----------

from functools import partial
from typing import Any

import numpy as np
from PIL import Image
from shutil import rmtree
from uuid import uuid4
from streaming import MDSWriter  # mosiacml-streaming
import pyspark.sql.functions as F
from pyspark.sql.column import Column
from pyspark.sql import DataFrame

import torchvision
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from streaming import StreamingDataset  # mosiacml-streaming

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel
from tsdb.ml.utils import OptimizerArgs
from tsdb.ml.yolo_trainer import YoloModelTrainer
from tsdb.ml.utils import OptimizerArgs

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

def convert_to_mds(
    df: DataFrame, columns: dict[str, str], compression: str, out_root: str, **kwargs
) -> None:
    """
    Function that converts a Spark DataFrame to a collection of `.mds` files.

    Args:
    df: The spark dataframe to be converted
    columns: A dictionary which contains the column names of the dataframe
            as keys and the corresponding mds data types as values
    compression: Compression algorithm name to use
    out_root: The local or remote directory path to store the output compressed files
    """
    pd_df = df.toPandas()
    samples = pd_df.to_dict("records")

    for sample in samples:
        sample["img"] = Image.open(sample["im_file"])
        sample["ori_shape"] = np.array(sample["img"].size, dtype=np.uint32)

    # Use `MDSWriter` to iterate through the input data and write to a collection of `.mds` files.
    # Note this has been unit tested here: https://github.com/mosaicml/streaming/blob/main/tests/test_writer.py
    with MDSWriter(
        out=out_root, columns=columns, compression=compression, **kwargs
    ) as out:
        for sample in samples:
            out.write(sample)


def collate_fn_img(data, transforms: callable) -> dict[str, Any]:
    """
    Function for collating data into batches. Some additional
    processing of the data is done in this function as well
    to get the batch into the format expected by the Ultralytics
    DetectionModel class.

    Args:
        data: The data to be collated a batch
        transforms: Torchvision transforms applied to the PIL images
    Returns: A dictionary containing the collated data in the formated 
            expected by the Ultralytics DetectionModel class
    """
    result = defaultdict(list)

    for i, element in enumerate(data):
        for key, value in element.items():
            if key == "img":
                value = transforms(value)

                _, h, w = value.shape  # h & w after sime augmentation has been don

                result["resized_shape"].append((h, w))
                result["ratio_pad"].append(
                    (
                        result["resized_shape"][i][0] / element["ori_shape"][0],
                        result["resized_shape"][i][1] / element["ori_shape"][1],
                    )
                )  # from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py#L295

            result[key].append(value)

        num_boxes = len(
            element["cls"]
        )  # no need to use reshape b/c len of cls array tells us how many bboxes

        for _ in range(num_boxes):
            result["batch_idx"].append(
                float(i)
            )  # yolo dataloader has this as a float not int

    result["img"] = torch.stack(result["img"], dim=0)
    result["batch_idx"] = torch.tensor(result["batch_idx"])

    # Unit test reshaping
    result["bboxes"] = torch.tensor(np.concatenate(result["bboxes"]).reshape(-1, 4))
    result["cls"] = torch.tensor(
        np.concatenate(result["cls"], dtype=float).reshape(-1, 1)
    )

    result["im_file"] = tuple(result["im_file"])
    result["ori_shape"] = tuple(result["ori_shape"])
    result["resized_shape"] = tuple(result["resized_shape"])
    result["ratio_pad"] = tuple(result["ratio_pad"])

    return dict(result)


def get_dataloader(
    local_dir: str, remote_dir: str, batch_size: int, **kwargs
) -> DataLoader:
    """
    Function that creates a PyTorch DataLoader from a collection of `.mds` files.

    Args:
    local_dir: Local directory where dataset is cached during training
    remote_dir: The local or remote directory where dataset `.mds` files are stored
    batch_size: The batch size of the dataloader and dataset.
          See: https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/faqs_and_tips.html
    """

    # Note that StreamingDataset is unit tested here: https://github.com/mosaicml/streaming/blob/main/tests/test_streaming.py
    dataset = StreamingDataset(
        local=local_dir,
        remote=remote_dir,
        batch_size=batch_size,
        split=None,
        shuffle=True,
        **kwargs
    )

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    # Create PyTorch DataLoader
    collate_fn = partial(collate_fn_img, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader

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

local_dir = '/local/cache/path7'

batch_size = 64

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
