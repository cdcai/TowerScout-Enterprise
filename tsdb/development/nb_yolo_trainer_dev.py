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
    model.args.conf = 0.002 # Note that this isn't set in the default config file. We must set it ourselves
    #model.args.single_cls = True # This will need to be set to true for towerscout since we are doing single class detection

    return model

# COMMAND ----------

model = get_model(model_yaml, model_pt, data)

# COMMAND ----------

model.args.single_cls = True # set to False when using coco dataset since it has 80 classes

# COMMAND ----------

yolo_trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=model)

# COMMAND ----------

batch_size = 4

# seems like confusion matrix methods only work when the dataset is built with
# mode="val" or else the minibatches wont have the key "ratio_pad" that is needed by some functions
model.args.fraction = 1.0 # fraction of dataset to use when in train mode
dataset = build_yolo_dataset(
    cfg=model.args, data=data, img_path=data["train"], batch=batch_size, mode="train"
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

model.args.single_cls = True # set to False when using coco dataset since it has 80 classes

# COMMAND ----------

import torch
from torch import tensor
from tsdb.ml.yolo_trainer import _prepare_batch, _prepare_pred, score, postprocess
from ultralytics.utils.metrics import ConfusionMatrix, box_iou


shape = (2, 3, 1500, 1500)  # To create a random tensor (image) with the given shape
batch = {
    "im_file": (
        "path/img1.jpg",
        "path/img2.jpg",
    ),
    "ori_shape": ((1500, 1500), (1500, 1500)),
    "resized_shape": ((1500, 1500), (1500, 1500)),
    "ratio_pad": ( ((2.,), (1.1, 1.1)), ((2.,), (1.1, 1.1)) ),
    "img": torch.randint(0, 256, shape, dtype=torch.uint8),
    "cls": tensor([[0.0], [0.0], [0.0], [0.0], [0.0]]),  # class labels
    "bboxes": tensor(
        [
            [0.5746, 0.6375, 0.2610, 0.3689],
            [0.3660, 0.6481, 0.1675, 0.3164],
            [0.5915, 0.5939, 0.1315, 0.1461],
            [0.4127, 0.5856, 0.1139, 0.1259],
            [0.3695, 0.7020, 0.0239, 0.0671],
        ]
    ),
    "batch_idx": tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
}


pred_prepared = tensor([[211.1375, 366.8750, 336.7625, 604.1750, 0.9, 0.0],
        [393.7625, 390.0875, 492.3875, 499.6625, 0.77, 0.0],
        [266.2625, 391.4375, 351.6875, 485.8625, 0.6, 0.0],
        [267.6125, 500.7875, 285.5375, 551.112, 0.39, 0.00]]).unsqueeze(0)  # Adding batch dimension


pred_unprepared = pred_prepared.clone()
pred_unprepared[:, :, :4] += 1.1
pred_unprepared[:, :, :4] *= 2
"""
For score function this test case should return {'accuracy_VAL': 0.75, 'f1_VAL': 0.8571428571428571}
because the confusion matrix should be:
 [[          3           0]
 [          1           0]]

because the first 3 predicted bounding box classes are correct and while the 4th one is also right
it's confidence does not meet the required threshold so it gets filtered out in the postprocess
step leading to a false netagtive for the fourth bounding box [0.4127, 0.5856, 0.1139, 0.1259].
Note that there are 5 bounding boxes in the batch but we onl
"""

# COMMAND ----------

args = model.args
args.conf = 0.5

conf_mat = ConfusionMatrix(
        nc=1, task="detect", conf=args.conf, iou_thres=0.45
    )  # only 1 class: cooling towers

pred_postproccesed = postprocess(pred_unprepared, model.args, [])

for si, pred in enumerate(pred_postproccesed):
    # print("batch:", batch)
    #print("\npred:", pred)
    pbatch = _prepare_batch(si, batch, "cpu")
    #idx = batch["batch_idx"] == si
    #pbatch = {"bbox": batch["bboxes"][idx], "cls": batch["cls"][idx], 'imgsz': batch["img"].shape[2:], 'ori_shape': batch["ori_shape"][si], "ratio_pad": batch["ratio_pad"][si]}
    print("\npbatch:", pbatch)
    cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    #print("cls:", cls)
    predn = _prepare_pred(pred, pbatch)
    print("\npredn:", predn)
    #print("\nbox IoU:", box_iou(bbox, predn[:, :4]))
   
    conf_mat.process_batch(detections=predn, gt_bboxes=bbox, gt_cls=cls)
    #print(conf_mat.matrix.sum(1), conf_mat.conf, conf_mat.iou_thres)
    print("Confusion matrix:\n",conf_mat.matrix)
