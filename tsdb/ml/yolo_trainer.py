import ultralytics.utils as uutils
from ultralytics.nn.tasks import DetectionModel

import torch
from torch import nn  # nn.
from torch import Tensor  # torch.Tensor, torch.optim
from torch.utils.data import DataLoader
from typing import Union, Any
from enum import Enum, auto

from mlflow.models.signature import infer_signature, ModelSignature

from tsdb.ml.model_trainer import BaseTrainer, TrainingArgs
from tsdb.ml.utils import Steps


class YOLOLoss(Enum):
    """
    Enum for the different loss types for the YOLO model. BL corresponds to box loss, BCE correspnds to
    binary cross entropy, and DFL corresponds to Distribution Focal loss.
    For more info see: https://docs.ultralytics.com/reference/utils/loss/
    """

    BL = auto()
    BCE = auto()
    DFL = auto()


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
    # set this to None to have scale_boxes compute for us
    ratio_pad = None
    if len(cls):
        bbox = (
            uutils.ops.xywh2xyxy(bbox)
            * torch.tensor(imgsz, device=device)[[1, 0, 1, 0]]
        )  # target boxes
        uutils.ops.scale_boxes(
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
    uutils.ops.scale_boxes(
        pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=None
    )  # native-space pred
    return predn


def postprocess(
    preds: Tensor, args: uutils.IterableSimpleNamespace, lb: list[Tensor]
) -> Tensor:
    """
    Apply Non-maximum suppression to prediction outputs.
    This is a method from the DectionValidator class see: ultralytics/models/yolo/detect/val.py
    """
    return uutils.ops.non_max_suppression(
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
    args: uutils.IterableSimpleNamespace,
) -> dict[str, float]:  # pragma: no cover
    """
    Returns a dictionary of metrics to be logged.
    Code adapted from the update_metrics method in: ultralytics/models/yolo/detect/val.py
    NOTE: preds must be outputs from the model when it is in eval() mode NOT train() mode.
    """
    conf_mat = uutils.metrics.ConfusionMatrix(
        nc=1, task="detect", conf=args.conf
    )  # only 1 class: cooling towers

    height, width = minibatch["img"].shape[2:]
    nb = len(minibatch["img"])
    bboxes = minibatch["bboxes"] * torch.tensor(
        (width, height, width, height), device=device
    )

    def _concat(index):
        selected = minibatch["batch_idx"] == index
        items = [minibatch["cls"][selected], bboxes[selected]]
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
    # use [:-1] to exlcude background class since task=detect
    fn = conf_mat.matrix.sum(0)[:-1] - tp
    tp, fp, fn = tp[0], fp[0], fn[0]
    tn = N - (tp + fp + fn)
    acc = (tp + tn) / (tp + fp + fn + tn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    # adding small constant for stability to avoid div by 0
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn)
    metrics = {
        f"accuracy_{step}": acc,
        f"f1_{step}": f1,
        f"recall_{step}": recall,
        f"precision_{step}": precision,
    }

    return metrics


@torch.no_grad()
def inference_step(
    minibatch: dict[str, Union[Tensor, int, float, str]],
    model: DetectionModel,
    step: str,
    device: str,
) -> dict[str, float]:
    model.eval()
    # for inference (non-dict input) ultralytics forward implementation returns a tensor not the loss
    pred = model(minibatch["img"])
    return score(minibatch, pred, step, device, model.args)


class YoloModelTrainer(BaseTrainer):
    """
    Model trainer class for the YOLO object detection model (DetectionModel class) from Ultralytics.
    Note that we have removed the torch.nn.parallel.DistributedDataParallel (DDP) usage that was present in 
    Ultralytics since we use Hyperopt for distributed tuning and it's not clear how nicely they will interact with each other.
    """

    def __init__(
        self,
        model: DetectionModel,
        optimizer: torch.optim.Optimizer = None,
        train_args: TrainingArgs = None,
        epochs: int = 1,
        **kwargs
    ):  # pragma: no cover
        super().__init__(model, optimizer, train_args, epochs)
        self.freeze_layers()
        self.loss_types = [loss.name for loss in YOLOLoss]

    def freeze_layers(self, freeze=None):
        """
        Freezes layers of YOLO, always freezes dfl layers
        """
        # Freeze layers
        freeze_list = (
            freeze
            if isinstance(freeze, list)
            else range(freeze)
            if isinstance(freeze, int)
            else []
        )

        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                # LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                # LOGGER.info(
                #     f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                #     "See ultralytics.engine.trainer for customization of frozen layers."
                # )
                v.requires_grad = True

    def preprocess_train(
        self, batch: dict[str, Union[Tensor, int, float, str]]
    ):  # pragma: no cover
        """
        Preprocesses a batch of images by scaling and converting to float.
        Code adapted from preprocess_batch method in: ultralytics/models/yolo/detect/train.py
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        # didn't include self.args.multi_scale if statement from source code
        return batch

    def preprocess_val(
        self, batch: dict[str, Union[Tensor, int, float, str]]
    ):  # pragma: no cover
        """
        Preprocesses a batch of images for validation.
        Code adapted from preprocess method in: ultralytics/models/yolo/detect/val.py
        Note: We didn't include the self.args.multi_scale if statement from the source code
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (
            batch["img"].half() if self.args.half else batch["img"].float()
        ) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch

    def training_step(self, minibatch: Union[Tensor, int, float, str]) -> dict:
        """
        TODO: test this
        """
        self.model.train()

        with uutils.torch_utils.autocast(self.amp):
            # forward pass
            minibatch = self.preprocess_train(minibatch)

            # Note: criterion is implemented as a class in ultralytics
            preds = self.model(
                minibatch["img"], augment=False
            )  # can also get loss directly by passing whole dict

            # Note: we are not including tloss variable b/c it seems to only be used for logging purposes
            self.loss, loss_items = self.model.loss(batch=minibatch, preds=preds)

        loss_scores = {
            f"{name}_{Steps.TRAIN.name}": loss_items[i].item()
            for i, name in enumerate(self.loss_types)
        }

        loss_scores["loss"] = self.loss
        
        return loss_scores

    @torch.no_grad()
    def validation_step(
        self, minibatch: Union[Tensor, int, float, str], step: Steps = Steps.VAL
    ) -> dict:  # pragma: no cover
        minibatch = self.preprocess_val(minibatch)
        # NOTE: moved inference_step logic into this function, added no_grad decorator to this
        self.model.eval()
        # for inference (non-dict input) ultralytics forward implementation returns a tensor not the loss
        preds = self.model(minibatch["img"])
        metrics = score(minibatch, preds, step.name, self.device, self.model.args)

        if step.name == "VAL":
            loss, loss_items = self.model.loss(batch=minibatch, preds=preds)
            metrics["loss_VAL"] = loss.cpu().item()
            loss_scores = {
                f"{name}_{step.name}": loss_items[i].item()
                for i, name in enumerate(self.loss_types)
            }
            metrics = {**metrics, **loss_scores}

        return metrics

    def get_signature(self, dataloader: DataLoader) -> ModelSignature:
        """
        Returns the mlflow signature of the model for logging and registration
        """
        self.model.eval()
        minibatch = next(iter(dataloader))
        minibatch = self.preprocess_val(minibatch)

        # Note: we are using the first image in the batch
        signature = infer_signature(
            model_input=minibatch["img"][0].cpu().numpy(),
            model_output=self.model(minibatch["img"])[0].detach().cpu().numpy(),
        )

        return signature
