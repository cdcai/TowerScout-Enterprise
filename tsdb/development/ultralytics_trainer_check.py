# Databricks notebook source
# import shutil
# import os
# from pyspark.sql.functions import udf
# from pyspark.sql.types import StringType
# import numpy as np
# from ultralytics.utils.ops import xyxy2xywh

# def copy_file(image_path, split_label, bboxes):
#     img_dir = label_dir = "/Volumes/edav_dev_csels/towerscout/data/ultralytics_training/images/"
#     label_dir = "/Volumes/edav_dev_csels/towerscout/data/ultralytics_training/labels/"

#     image_name = image_path.split("/")[-1]  # get image name
#     image_name = image_name.split(".")[0]  # remove extension
    
#     # check if directory exists
#     if not os.path.exists(f"{label_dir}/{split_label}"): 
#         os.makedirs(f"{label_dir}/{split_label}")

#     # # copy image to appropraite directory
#     # img_dest = f"{img_dir}/{split_label}/"
#     # try:
#     #     shutil.copy(image_path, img_dest)
#     # except:
#     #     return "Error copying file {image_path}"

#     # create corresponding txt file with class label and bbox coords
#     label_file = f"{label_dir}/{split_label}/{image_name}.txt"
#     with open(label_file, 'w') as f:
#         for bbox in bboxes:
#             bbox = np.array([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
#             bbox = xyxy2xywh(bbox)
#             f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

#     return f"Copied {image_path}"

# # Register the UDF with Spark
# copy_file_udf = udf(copy_file, StringType())

# # create directory structure
# df = spark.read.table("edav_dev_csels.towerscout.test_image_gold") #.limit(40)

# # Apply the UDF to the DataFrame
# df = df.withColumn("copy_status", copy_file_udf(df["image_path"], df["split_label"], df["bboxes"]))

# # Show the results
# display(df)

# COMMAND ----------

from ultralytics import YOLO
import ultralytics.engine.trainer as trainer
import mlflow

mlflow.end_run()

mlflow.autolog()

mlflow.start_run()

# Load a model
model = YOLO("yolov10n.yaml").load("yolov10n.pt")
model.args['patience'] = 50

# see: https://github.com/ultralytics/ultralytics/issues/16446#issuecomment-2372304865
trainer.RANK = -1
trainer.LOCAL_RANK = -1

# Train the model
results = model.train(data="cooling_towers.yaml", epochs=300, imgsz=640)
# results = model.tune(data="cooling_towers.yaml", epochs=10, iterations=10, optimizer="AdamW")
mlflow.end_run()

# COMMAND ----------

mlflow.end_run()
