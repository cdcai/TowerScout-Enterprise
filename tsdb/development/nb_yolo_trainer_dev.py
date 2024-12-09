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

import torch 
shape = (2, 3, 640, 640) # Create a random tensor with the given shape
random_image_tensor = torch.randint(0, 256, shape, dtype=torch.uint8)

display(random_image_tensor.shape)

# COMMAND ----------

model.args.single_cls = True # set to False when using coco dataset since it has 80 classes

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
batch_size = 2

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


