from ultralytics.utils.torch_utils import TORCH_2_4, autocast
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils import ops
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.nn.tasks import DetectionModel

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from typing import Union
from enum import Enum, auto

from mlflow.models.signature import infer_signature, ModelSignature

from tsdb.ml.utils import OptimizerArgs, Steps, TrainingArgs

class YOLOLoss(Enum):
    """
    Enum for the different loss types for the YOLO model. BL corresponds to box loss, BCE correspnds to 
    binary cross entropy, and DLF corresponds to Distribution Focal loss.
    For more info see: https://docs.ultralytics.com/reference/utils/loss/
    """
    BL = auto()
    BCE = auto()
    DLF = auto()

def _prepare_batch(
    si: int, batch: Tensor, device: str
) -> dict[str, Union[Tensor, int, float, str]]:
    """
    Prepares a batch of images and annotations for validation.
    This is a method from the DectionValidator class see: ultralytics/models/yolo/detect/val.py
    """
    idx = batch["batch_idx"] == si
    cls = batch["cls"][idx].squeeze(-1)
    bbox = batch["bboxes"][idx]
    ori_shape = batch["ori_shape"][si]
    imgsz = batch["img"].shape[2:]
    try:  # this key only exists if dataloader is a val dataloader
        ratio_pad = batch["ratio_pad"][si]
    except:
        ratio_pad = None
    if len(cls):
        bbox = (
            ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=device)[[1, 0, 1, 0]]
        )  # target boxes
        ops.scale_boxes(
            imgsz, bbox, ori_shape, ratio_pad=ratio_pad
        )  # native-space labels
    return {
        "cls": cls,
        "bbox": bbox,
        "ori_shape": ori_shape,
        "imgsz": imgsz,
        "ratio_pad": ratio_pad,
    }


def _prepare_pred(
    pred: Tensor, pbatch: dict[str, Union[Tensor, int, float, str]]
) -> Tensor:
    """
    Prepares a batch of images and annotations for validation.
    This is a method from the DectionValidator class see: ultralytics/models/yolo/detect/val.py
    """
    predn = pred.clone()
    ops.scale_boxes(
        pbatch["imgsz"],
        predn[:, :4],
        pbatch["ori_shape"],
        ratio_pad=pbatch["ratio_pad"],
    )  # native-space pred
    return predn


def postprocess(
    preds: Tensor, args: IterableSimpleNamespace, lb: list[Tensor]
) -> Tensor:
    """
    Apply Non-maximum suppression to prediction outputs.
    This is a method from the DectionValidator class see: ultralytics/models/yolo/detect/val.py
    """
    return ops.non_max_suppression(
        preds,
        args.conf,
        args.iou,
        labels=lb,
        multi_label=True,
        agnostic=args.single_cls or args.agnostic_nms,
        max_det=args.max_det,
    )


def score(
    minibatch: dict[str, Union[Tensor, int, float, str]],
    preds: Tensor,
    step: str,
    device: str,
    args: IterableSimpleNamespace,
) -> dict[str, float]:  # pragma: no cover
    """
    Returns a dictionary of metrics to be logged.
    Code adapted from the update_metrics method in: ultralytics/models/yolo/detect/val.py
    NOTE: preds must be outputs from the model when it is in eval() mode NOT train() mode.
    """
    conf_mat = ConfusionMatrix(nc=1, task="detect")  # only 1 class: cooling towers

    height, width = minibatch["img"].shape[2:]
    nb = len(minibatch["img"])
    bboxes = minibatch["bboxes"] * torch.tensor(
        (width, height, width, height), device=device
    )

    def _concat(index):
        selected = minibatch["batch_idx"] == index
        items = [
            minibatch["cls"][selected],
            bboxes[selected]
        ]
        return torch.cat(items, dim=-1)
    
    if args.save_hybrid:
        lb = [_concat(i) for i in range(nb)]
    else:
        lb = []

    preds = postprocess(preds, args, lb)
    for si, pred in enumerate(preds):
        pbatch = _prepare_batch(si, minibatch, device)
        cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
        nl = len(cls)
        npr = len(pred)  # num predictions
        if npr == 0:
            if nl:
                conf_mat.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

        if args.single_cls:
            pred[:, 5] = 0

        predn = _prepare_pred(pred, pbatch)
        conf_mat.process_batch(predn, bbox, cls)

    N = conf_mat.matrix.sum()  # total number of instances
    tp, fp = conf_mat.tp_fp()  # returns list of tp & fp per class
    fn = conf_mat.matrix.sum(0)[:-1] - tp  # use [:-1] to exlcude background class since task=detect
    tp, fp, fn = tp[0], fp[0], fn[0]
    tn = N - (tp + fp + fn)
    acc = (tp + tn) / (tp + fp + fn + tn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    metrics = {f"accuracy_{step}": acc, f"f1_{step}": f1}

    return metrics


@torch.no_grad()
def inference_step(
    minibatch: dict[str, Union[Tensor, int, float, str]],
    model: DetectionModel,
    step: str,
    device: str,
) -> dict[str, float]:
    
    model.eval()
    pred = model(minibatch["img"])  # for inference (non-dict input) ultralytics forward implementation returns a tensor not the loss
    return score(minibatch, pred, step, device, model.args)


"""
Model trainer class for the YOLO object detection model (DetectionModel class) from Ultralytics.
Note that we have removed the torch.nn.parallel.DistributedDataParallel (DDP) usage that was present in 
Ultralytics since we use Hyperopt for distributed tuning and it's not clear how nicely they will interact with each other.
"""

class YoloModelTrainer:
    def __init__(self, optimizer_args: OptimizerArgs, model: DetectionModel, train_args: TrainingArgs):  # pragma: no cover
        self.model = model
        self.train_args = train_args
        self.args = self.model.args
        self.amp = self.args.amp
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp)
            if TORCH_2_4
            else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        self.optimizer = self.build_optimizer(
            model=self.model,
            name=optimizer_args.optimizer_name,
            lr=optimizer_args.lr0,
            momentum=optimizer_args.momentum,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.loss_types = [loss.name for loss in YOLOLoss]

    def build_optimizer(
        self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5
    ):
        """
        Code adapted from the build_optimizer method in: ultralytics/models/yolo/train.py
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        
        # TODO: test this
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(
                0.002 * 5 / (4 + nc), 6
            )  # lr0 fit equation to 6 decimal places
            name, lr, momentum = (
                ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            )
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)

        return optimizer

    def preprocess_train(self, batch: dict[str, Union[Tensor, int, float, str]]):  # pragma: no cover
        """
        Preprocesses a batch of images by scaling and converting to float.
        Code adapted from preprocess_batch method in: ultralytics/models/yolo/detect/train.py 
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        # didn't include self.args.multi_scale if statement from source code
        return batch

    def preprocess_val(self, batch: dict[str, Union[Tensor, int, float, str]]):  # pragma: no cover
        """
        Preprocesses a batch of images for validation.
        Code adapted from preprocess method in: ultralytics/models/yolo/detect/val.py 
        Note: We didn't include the self.args.multi_scale if statement from the source code
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch

    def optimizer_step(self):  # pragma: no cover
        """
        Perform a single step of the training optimizer with gradient clipping and EMA update.
        Note: We didn't include the self.ema conditional from the source code
        """
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def training_step(
        self, minibatch: Union[Tensor, int, float, str], **kwargs
    ) -> dict:
        """
        TODO: test this
        """
        self.model.train()

        # forward pass
        minibatch = self.preprocess_train(minibatch)

        # Note: criterion is implemented as a class in ultralytics
        preds = self.model(minibatch["img"], augment=False)  # can also get loss directly by passing whole dict
        
        # Note: we are not including tloss variable b/c it seems to only be used for logging purposes
        self.loss, loss_items = self.model.loss(batch=minibatch, preds=preds)

        # backward pass
        self.scaler.scale(self.loss).backward()

        # Note: in ultralytics, losses are accumulated between N batchs before calling optimizer_step
        self.optimizer_step()

        loss_scores = {
            f"{name}_{Steps['TRAIN'].name}": loss_items[i].item()
            for i, name in enumerate(self.loss_types)
        }

        return loss_scores

    def validation_step(
        self, minibatch: Union[Tensor, int, float, str], **kwargs
    ) -> dict:  # pragma: no cover
        minibatch = self.preprocess_val(minibatch)
        return inference_step(minibatch, self.model, Steps["VAL"].name, self.device)
    
    def get_signature(self, dataloader: DataLoader) -> ModelSignature:
        """
        Returns the mlflow signature of the model for logging and registration
        """
        self.model.eval()
        dataloader_iter = iter(dataloader)
        minibatch = next(data_iter)
        minibatch = self.preprocess_val(minibatch)

        signature = infer_signature(
                model_input=minibatch["img"][0].numpy(), # Note: we are using the first image in the batch
                model_output=self.model(minibatch["img"]).detach().numpy(),
            ) 
        
        return signature

    def save_model(self) -> None:  # pragma: no cover
        pass

