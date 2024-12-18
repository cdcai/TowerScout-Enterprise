# Databricks notebook source
# MAGIC %md
# MAGIC # Previewing Bounding Boxes
# MAGIC During the inference pipeline process, we needed to review the bounding boxes created by the model. Below is the code used to do this task. This may be useful for future uses, so here it is documented.

# COMMAND ----------

from typing import TypedDict

from torch import tensor
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

# COMMAND ----------

class BoundingBox(TypedDict):
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



# COMMAND ----------

user_id = "cnu4"
request_id = "d55aa5c6"
df = (
    spark
    .read
    .format("delta")
    .table("edav_dev_csels.towerscout.test_image_silver")
    .filter(f"user_id = '{user_id}' AND request_id = '{request_id}'")
).toPandas()

display(df)

# COMMAND ----------

image_index = 1
selected_image = df.iloc[image_index]

image = read_image(selected_image["image_path"].lstrip("dbfs:"))
bboxes = tensor(
    [denormalize_bounding_boxes(item) for item in selected_image["bboxes"]]
)
test_image = draw_bounding_boxes(image, bboxes, colors="red")
display(to_pil_image(test_image))

# COMMAND ----------


