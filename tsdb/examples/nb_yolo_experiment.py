# Databricks notebook source
# MAGIC %pip install opencv-python # need for yolo model
# MAGIC %pip install ultralytics==8.2.92 # need for yolo model
# MAGIC %pip install gitpython==3.1.30 pillow==10.3.0 requests==2.32.0 setuptools==70.0.0 # need for loading yolo with torch.hub.load from ultralytics

# COMMAND ----------

import os
from IPython.display import Image
import torch

# COMMAND ----------

splits = ["train", "val", "test"]

for split in splits:
    images_path = f"/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_data/images/{split}/"
    labels_path = f"/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_data/labels/{split}/"

    img_files = os.listdir(images_path)
    labels_files = os.listdir(labels_path)
    classes = {}
    singeltons = 0
    for labels_file_path in labels_files:
        with open(labels_path + labels_file_path, "r") as file:
            for line in file:
                bboxs = line.strip().split(" ")
                if len(bboxs) > 1:
                    label, _, _, _, _ = bboxs
                else:
                    singeltons += 1
                    continue
                try:
                    classes[label] += 1
                except:
                    classes[label] = 1

    print(f"{singeltons} singelton lines encoutered and skipped")
    print(f"Unique classes and counts in the {split} set: {classes}")
    print(
        f"{len(img_files)} image files and {len(labels_files)} label files in {split} set. {len(img_files)-len(labels_files)} images with no cooling towers.\n"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Check performance of current YOLOv5 model

# COMMAND ----------

!python /Volumes/edav_dev_csels/towerscout_test_schema/ultralytics_yolov5_master/val.py --weights /Volumes/edav_dev_csels/towerscout_test_schema/test_volume/model_params/yolo/xl_250_best.pt --data /Volumes/edav_dev_csels/towerscout_test_schema/towerscout_data/data.yaml --img 640 --project . --name yolov5_val

# COMMAND ----------

display(Image(filename="yolov5_val/val_batch0_pred.jpg"))

# COMMAND ----------

display(Image(filename="yolov5_val/val_batch0_labels.jpg"))

# COMMAND ----------

display(Image(filename="yolov5_val/confusion_matrix.png"))

# COMMAND ----------

display(Image(filename="yolov5_val/F1_curve.png"))

# COMMAND ----------

# MAGIC %md
# MAGIC # YOLOv5 model experiements

# COMMAND ----------

exp_name = "yolo5v_exp"

# COMMAND ----------

!python yolo5v_train.py 32 50 yolo5v_exp

# COMMAND ----------

display(Image(filename=f"runs/detect/{exp_name}/confusion_matrix.png"))

# COMMAND ----------

display(Image(filename=f"runs/detect/{exp_name}/F1_curve.png"))

# COMMAND ----------

display(Image(filename=f"runs/detect/{exp_name}/results.png"))
