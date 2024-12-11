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
    #print(image_batch)
    #print(len(yolo_trainer.model(image_batch["img"], augment=False)))
    print(f"Training metrics: {metrics}")


for image_batch in loader:
    metrics = yolo_trainer.validation_step(image_batch)
    yolo_trainer.model.eval()
    print(yolo_trainer.model(image_batch["img"])[0].shape)
    print(type(yolo_trainer.model))
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
        [267.6125, 500.7875, 285.5375, 551.112, 0.39, 0.05]]).unsqueeze(0)  # Adding batch dimension


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

# COMMAND ----------


score(
    minibatch = batch,
    preds = pred_unprepared,
    step = "VAL",
    device = "cpu",
    args = model.args
)

# COMMAND ----------

import torch
from torch import tensor
from tsdb.ml.yolo_trainer import _prepare_batch

batch = {
    "im_file": (
        "path/img1.jpg",
        "path/img2.jpg",
    ),
    "ori_shape": ((500, 381), (478, 640)),
    "resized_shape": ((640, 640), (640, 640)),
    "ratio_pad": (((1.28, 1.2808398950131235), (76, 0)), ((1.0, 1.0), (0, 81))),
    "img": torch.randint(0, 256, shape, dtype=torch.uint8),
    "cls": tensor(
        [[17.0], [17.0], [0.0], [0.0], [58.0]]
    ),
    "bboxes": tensor(
        [
            [0.5746, 0.6375, 0.2610, 0.3689],
            [0.3660, 0.6481, 0.1675, 0.3164],
            [0.5915, 0.5939, 0.1315, 0.1461],
            [0.4127, 0.5856, 0.1139, 0.1259],
            [0.3695, 0.7020, 0.0239, 0.0671]
        ]
    ),
    "batch_idx": tensor([0.0, 1.0, 0.0, 1.0, 1.0]) # batch_idx corresponds to index of the image the box is from in the im_file section of this dict
}

# COMMAND ----------

pbatch = _prepare_batch(1, batch, 'cpu')
#print(pbatch)
assert torch.equal(pbatch['cls'], tensor([17., 0., 58.]))
assert pbatch['ratio_pad'] == batch['ratio_pad'][1]

# COMMAND ----------

from tsdb.ml.yolo_trainer import _prepare_batch
from ultralytics.cfg import get_cfg

data_yaml_path = "coco8.yaml"
data = check_det_dataset(data_yaml_path)
args = get_cfg()
batch_size = 4

dataset = build_yolo_dataset(
    cfg=args, data=data, img_path=data["val"], batch=batch_size, mode="val"
)

loader = build_dataloader(dataset, batch_size, workers=4)

for i, image_batch in enumerate(loader):
    #print(f"{len(image_batch['img'])} images in this batch.") # batch of 4 imgs
    print(image_batch['ratio_pad'])
    #print(image_batch["batch_idx"] == si)
    #pbatch = _prepare_batch(i, image_batch, 'cpu')
    #print(pbatch)



# COMMAND ----------

from tsdb.ml.yolo_trainer import _prepare_pred
import torch

img1_shape = (1280, 720)
img2_shape = (640, 360)

input2 = {"imgsz": img1_shape, "ori_shape": img2_shape, "ratio_pad": ((1.0,), (0, 0))}

boxes = torch.tensor([[100, 200, 300, 400], [50, 60, 150, 160]], dtype=torch.float32)

expected_boxes = torch.tensor([[200, 400, 600, 800], [100, 120, 300, 320]], dtype=torch.float32)

predn = _prepare_pred(boxes, input2)

torch.allclose(predn, expected_boxes, atol=1e-3)

# COMMAND ----------

import pytest
import torch
from ultralytics.utils.ops import scale_boxes, clip_boxes
from tsdb.ml.yolo_trainer import _prepare_pred

@pytest.mark.parametrize(
    "img1_shape, img0_shape, boxes, ratio_pad, expected_boxes",
    [
        (
            (1280, 720),
            (640, 360),
            torch.tensor(
                [[100, 200, 300, 400], [50, 60, 150, 160]], dtype=torch.float32
            ),
            ((1.0,), (0, 0)),
            torch.tensor(
                [[100, 200, 300, 400], [50, 60, 150, 160]], dtype=torch.float32
            ),
        )
    ],
)


def test_scale_boxes(img1_shape, img0_shape, boxes, ratio_pad, expected_boxes):
    scaled_boxes = scale_boxes(
        img1_shape, boxes.clone(), img0_shape, ratio_pad=ratio_pad
    )
    print(scaled_boxes)
    print(expected_boxes)
    assert torch.allclose(scaled_boxes, expected_boxes, atol=1e-3)


test_scale_boxes(
    (1280, 720),
    (640, 360),
    torch.tensor([[100, 200, 300, 400], [50, 60, 150, 160]], dtype=torch.float32),
    ((10.0,), (5, 5)),
    (torch.tensor([[100, 200, 300, 400], [50, 60, 150, 160]], dtype=torch.float32) - 5) / 10,
)

# COMMAND ----------

pred = tensor(
        [
            [0.508, 0.141, 0.27, 0.4, 0.9, 0],  # Box 1 (keep)
            [0.31, -0.42, 0.443, 0.5, 0.6, 0],  # Box 2 (keep)
            [0.92, 0.442, -0.43, 0.5, 0.46, 0],  # Box 3 (filter)
            [0.444, 0.2, 0.3, 0.5, 0.39, 0],  # Box 4 (filter)
        ]
    ).unsqueeze(0)#torch.tensor([[100.0, 200.0, 300.0, 400.0], [50.0, 60.0, 150.0, 160.0]])
pbatch_info = {"imgsz": (140, 140), "ratio_pad": ((1.0,), (0.1, 0.1)), "ori_shape": (140, 140) }
out = _prepare_pred(pred, pbatch_info)
print(out)
print((pred - 0.1)/1)

# COMMAND ----------

prediction = tensor([[180., 313., 287., 516., 0.9, 0.0],
        [336., 333., 420., 426., 0.77, 0.0],
        [227., 334., 300., 415., 0.6, 0.0],
        [228., 427., 244., 470., 0.39, 0.0]]).unsqueeze(0)  # Adding batch dimension

pbatch_info = {"imgsz": (1500, 1500), "ratio_pad": ((2.,), (1.1, 1.1)), "ori_shape": (1500, 1500) }

pred = prediction.clone()
pred[:, :, :4] += 1.1/2
pred[:, :, :4] *= 2
ratio_pad = pbatch_info["ratio_pad"]
gain = ratio_pad[0][0]
pad = ratio_pad[1]
print(gain, pad)
print(prediction)
print("")
print(_prepare_pred(pred, pbatch_info))

# COMMAND ----------

import torch

# Define your tensor
prediction = torch.tensor([
    [180.6400, 313.5360, 287.8400, 516.0320, 0.9, 0.0],
    [336.4800, 333.3440, 420.6400, 426.8480, 0.77, 0.0],
    [227.6800, 334.4960, 300.5760, 415.0720, 0.6, 0.0],
    [228.8320, 427.8080, 244.1280, 470.7520, 0.39, 0.0]
]).unsqueeze(0)

# Add 3 to the first 4 columns of each row
prediction[:, :, :4] += 3

# Print the full tensor
print(prediction)

