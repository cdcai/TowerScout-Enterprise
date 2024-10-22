# Databricks notebook source
# MAGIC %pip install ultralytics

# COMMAND ----------

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel
from tsdb.ml.utils import OptimizerArgs
from tsdb.ml.yolo_trainer import YoloModelTrainer

# COMMAND ----------

optimizer_args = OptimizerArgs(optimizer_name="Adam", lr0=0.002, momentum=0.9)

# COMMAND ----------

data_yaml_path = "/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_data/data.yaml" #"coco8.yaml"
data = check_det_dataset(data_yaml_path)
print(data)

model_yaml = "yolov8n.yaml"
model_pt = "yolov8n.pt"

def get_model(model_yaml, model_pt, data):
    """
    See DetectionTrainer class and BaseTrainer class for details on how to setup the model
    """
    args = get_cfg()  # used to get hyperparams for model and other stuff from some config file
    model = DetectionModel(cfg=model_yaml, verbose=False)
    weights, _ = attempt_load_one_weight(model_pt)
    model.load(weights)
    model.nc = data["nc"]  # attach number of classes to model
    model.names = data["names"]  # attach class names to model
    model.args = args
    model.args.conf = 0.001 # Note that this isn't set in the default config file. We must set it ourselves
    model.args.single_cls = True # This will need to be set to true for towerscout since we are doing single class detection

    return model

# COMMAND ----------

model = get_model(model_yaml, model_pt, data)

# COMMAND ----------

yolo_trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=model)

# COMMAND ----------

batch_size = 32

# seems like confusion matrix methods only work when the dataset is built with
# mode="val" or else the minibatches wont have the key "ratio_pad" that is needed in some methods
model.args.fraction = 0.00025 # fraction of dataset to use
dataset = build_yolo_dataset(
    cfg=model.args, data=data, img_path=data["val"], batch=batch_size, mode="val"
)

# COMMAND ----------

loader = build_dataloader(dataset, batch_size, workers=4)

# COMMAND ----------

for image_batch in loader:
    metrics = yolo_trainer.training_step(image_batch)
    print(f"Training metrics: {metrics}")

for image_batch in loader:
    metrics = yolo_trainer.validation_step(image_batch)
    print(f"Validation metrics: {metrics}")
