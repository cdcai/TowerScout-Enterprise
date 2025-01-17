from copy import copy

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
from tsdb.ml.yolo_validator import ModifiedDetectionValidator


class YOLOLoss(Enum):
    """
    Enum for the different loss types for the YOLO model. BL corresponds to box loss, BCE correspnds to
    binary cross entropy, and DFL corresponds to Distribution Focal loss.
    For more info see: https://docs.ultralytics.com/reference/utils/loss/
    """

    BL = auto()
    BCE = auto()
    DFL = auto()


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
        **kwargs,
    ):  # pragma: no cover
        super().__init__(model, optimizer, train_args, epochs, **kwargs)
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
            elif (
                not v.requires_grad and v.dtype.is_floating_point
            ):  # only floating point Tensor can require gradients
                # LOGGER.info(
                #     f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                #     "See ultralytics.engine.trainer for customization of frozen layers."
                # )
                v.requires_grad = True

    @staticmethod
    def get_validator(
        dataloader: DataLoader,
        training: bool,
        device: str,
        args: uutils.IterableSimpleNamespace,
    ) -> ModifiedDetectionValidator:
        """
        Returns a validator for performing validation on the model.
        This was made a static method to allow access to a validator
        when working with the test set so no model trainer is needed.

        Args:
        dataloader: The dataloader for the validation/test dataset
        training: Whether or not we are training the model
        device: The device to run the model on
        """
        return ModifiedDetectionValidator(
            dataloader, args=copy(args), training=training, device=device
        )

    def label_loss_items(self, loss_items: torch.Tensor, step: str = "VAL") -> dict[str, float] | list[str]:
        """
        Returns a loss dict with labelled training loss items tensor.
        Not needed for classification but necessary for segmentation & detection.
        Taken from:
        https://github.com/ultralytics/ultralytics/blob/09a34b19eddda5f1a92f1855b1f25f036300d9a1/ultralytics/engine/trainer.py#L628

        Args:
            loss_items: A tensor containing the loss items
            step: the prefix to prepend to the loss name/key

        Returns:
        loss_dict if loss_items is not None, else a list of loss names
        """
        keys = [f"{step}/{x}" for x in self.loss_types]
        if loss_items is not None:
            loss_items = [float(x) for x in loss_items]
            loss_dict = dict(zip(keys, loss_items))
            return loss_dict
        else:
            return keys

    def preprocess_train(
        self, batch: dict[str, Union[Tensor, int, float, str]]
    ):  # pragma: no cover
        """
        Preprocesses a batch of images by scaling and converting to float.
        Code adapted from preprocess_batch method in: ultralytics/models/yolo/detect/train.py
        Note: We didn't include the self.args.multi_scale if statement from the source code
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def preprocess_val(
        self, batch: dict[str, Union[Tensor, int, float, str]]
    ):  # pragma: no cover
        """
        Preprocesses a batch of images for validation.
        Code adapted from preprocess method in: ultralytics/models/yolo/detect/val.py
        Note: We didn't include the self.args.save_hybrid if statement from the source code
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

            # Note: criterion is implemented as a class in Ultralytics
            preds = self.model(
                minibatch["img"]
            )  # can also get loss directly by passing whole dict

            # Note: we are not including tloss variable b/c it seems to only be used for logging purposes
            self.loss, loss_items = self.model.loss(batch=minibatch, preds=preds)

        loss_scores = {
            f"{Steps.TRAIN.name}/{name}": loss_items[i].item()
            for i, name in enumerate(self.loss_types)
        }

        loss_scores["loss"] = self.loss

        return loss_scores, loss_items

    @torch.no_grad()
    def validation_step(
        self, loss_items: torch.Tensor, step: str = "VAL", **kwargs
    ) -> dict:  # pragma: no cover
        metrics, val_loss = self.validator(self.model, loss_items)
        num_batches = len(self.validator.dataloader)
        losses = self.label_loss_items(val_loss.cpu() / num_batches, step)
        metrics = {**metrics, **losses}
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