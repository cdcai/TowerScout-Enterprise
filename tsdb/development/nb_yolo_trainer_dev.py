# Databricks notebook source
# MAGIC %pip install ultralytics
# MAGIC %pip install efficientnet_pytorch

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

data_yaml_path = "coco8.yaml"
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
# mode="val" or else the minibatches wont have the key "ratio_pad" that is needed by some functions
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

# COMMAND ----------

from tsdb.ml.yolo_trainer import postprocess
import torch

for image_batch in loader:
    print(len(image_batch['img'])) # batch of 4 imgs
    proccessed = yolo_trainer.preprocess_val(image_batch)
    model.eval()
    preds = model(proccessed["img"]) # YOLOv8 model in validation model, output = (inference_out, loss_out)
    preds = preds[0]
    print("preds.shape:", preds.shape)
    print("preds[0].shape:", preds[0].shape)
    postprocessed = postprocess(preds, model.args, [])
    print(postprocessed[3])
    #test_input = (torch.tensor([1,2,3,4]), torch.tensor([1,2,3,4]))
    #print(postprocess(test_input, model.args, []))

# COMMAND ----------

import torch
#from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils import IterableSimpleNamespace
from tsdb.ml.yolo_trainer import postprocess

def sample_prediction():

    # box format: (x1, y1, x2, y2, confidence, class)
    prediction = torch.tensor([
                [0.508, 0.141, 0.27, 0.4, 0.9, 0],  # Box 1 (keep)
                [0.31, -0.42, 0.443, 0.5, 0.6, 0],  # Box 2 (keep)
                [0.92, 0.442, -0.43, 0.5, 0.46, 0],  # Box 3 (filter)
                [0.444, 0.2, 0.3, 0.5, 0.39, 0],  # Box 4 (filter)
            ]).unsqueeze(0) # Adding batch dimension
    
    return prediction


sample_prediction = sample_prediction()


args = IterableSimpleNamespace(conf=0.5, iou=0.0, single_cls=True, max_det=300)
lb = []
# Test with default thresholds and no classes specified
output = postprocess(sample_prediction, args, lb)

assert isinstance(output, list), "Output should be a list"
assert (
    len(output) == sample_prediction.shape[0]
), "Output list length should match batch size"

for out in output:
    assert isinstance(out, torch.Tensor), "Each output element should be a tensor"
    if len(out) > 0:
        assert (
            out[:, 4] >= args.conf
        ).all(), "All confidences should be above or equal to the threshold"

        assert len(out) == 2, "Output should contain 50% of the original boxes (2)"

print("Simple test passed.")
