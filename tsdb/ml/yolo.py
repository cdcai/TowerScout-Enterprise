from typing import Union
from enum import Enum, auto
from copy import copy

import ultralytics
import ultralytics.utils as uutils
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel

import torch
from torch.utils.data import DataLoader

from mlflow.models.signature import infer_signature, ModelSignature

from tsdb.ml.model_trainer import BaseTrainer, TrainingArgs
from tsdb.ml.utils import Steps


class YoloVersions(Enum):  # pragma: no cover
    """
    Enum for names of different yolo models by version and size.
    The yaml files can be found at:
    https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models
    """
    yolov10n = auto()
    yolov10s = auto()
    yolov10m = auto()
    yolov10b = auto()
    yolov10l = auto()
    yolov10x = auto()
    yolov9t = auto()
    yolov9s = auto()
    yolov9m = auto()
    yolov9c = auto()
    yolov9e = auto()


class YOLOLoss(Enum):  # pragma: no cover
    """
    Enum for the different loss types for the YOLO model. BCE_loss correspnds to
    binary cross entropy loss, and DF_loss corresponds to Distribution Focal loss.
    For more info see: https://docs.ultralytics.com/reference/utils/loss/
    """

    box_loss = auto()
    BCE_loss = auto()
    DF_loss = auto()


class ModifiedDetectionValidator(DetectionValidator):
    """
    A modified DetectionValidator object that inherets from the DetectionValidator class from Ultralytics.
    We overload the init_metric method to remove the `val` variable creation and the __call__ method
    to remove componenets that are not relevant to our use case.
    NOTE: Validator object tested here: https://github.com/ultralytics/ultralytics/blob/main/tests/test_engine.py
    
    Args:
        dataloader: A PyTorch dataloader for validation/testing datasets.
        args: A IterableSimpleNamespace object containing the arguments for the model and training (if applicable).
        training: Wether the model is being trained or not. This determines some behaviour in __call__.
        device: The device to run the model on.
    """

    def __init__(self, dataloader: DataLoader, args: uutils.IterableSimpleNamespace, training: bool, device: str):  # pragma: no cover
        super().__init__(dataloader=dataloader, args=args)
        self.training = training
        self.device = device

    def init_metrics(self, model: torch.nn.Module) -> None:  # pragma: no cover
        """
        Initialize evaluation metrics for YOLO.
        We override this to remove `val` variable creation
        and is_coco/is_lvis checks.

        Args:
            model: The model object (a DetectionModel instance for detection with YOLO).
        """
        self.is_coco = False
        self.is_lvis = False
        self.class_map = list(range(1, len(model.names) + 1))
        self.args.save_json |= (
            self.args.val and (self.is_coco or self.is_lvis) and not self.training
        )  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.end2end = getattr(model, "end2end", False)
        self.confusion_matrix = uutils.metrics.ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    @torch.no_grad()
    def __call__(
        self, model: torch.nn.Module, loss_items: torch.Tensor = None
    ) -> dict[str, float] | tuple[dict[str, float], torch.Tensor]:  # pragma: no cover
        """
        Executes validation process, running inference on dataloader and computing performance metrics.
        Args:
            model: The model object (DetectionModel for YOLO).
            loss_items: The loss_items tensor returned by the forward pass of a DetectionModel (only needed for validation not testing).

        Returns:
            stats: A dictionary of performance metrics f1(B), recall(B), mAP50(B) etc.
            loss: The loss tensor returned by the forward pass of a DetectionModel during inference (only needed for validation not testing).

        """
        augment = self.args.augment and (not self.training)
        if self.training:
            # force FP16 val during training
            self.args.half = self.device != "cpu" and self.args.amp
            self.loss = torch.zeros_like(loss_items, device=self.device)

        model = model.half() if self.args.half else model.float()
        model.eval()

        dt = (
            uutils.ops.Profile(device=self.device),
            uutils.ops.Profile(device=self.device),
            uutils.ops.Profile(device=self.device),
            uutils.ops.Profile(device=self.device),
        )

        self.init_metrics(uutils.torch_utils.de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(self.dataloader):
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)

        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()

        # compute f1 score from precision and recall computed by Ultralytics method.
        # (B) means boxes. See:
        # https://github.com/ultralytics/ultralytics/issues/9043#issuecomment-2006767946
        recall = stats["metrics/recall(B)"]
        precision = stats["metrics/precision(B)"]
        stats["metrics/f1(B)"] = 2 * recall * precision / (recall + precision + 1e-16)

        # metric not needed, defined as weighted average of metrics we already retrun. See:
        # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L748
        stats.pop("fitness") 

        if self.training:
            model.float()
            return stats, self.loss
        else:
            return stats


class YoloModelTrainer(BaseTrainer):
    """
    Model trainer class for the YOLO object detection model (DetectionModel class) from Ultralytics.
    Note that we have removed the torch.nn.parallel.DistributedDataParallel (DDP) usage that was present in
    Ultralytics since we use Hyperopt for distributed tuning and it's not clear how nicely they will interact with each other.

    Args:
        model: The DetectionModel instance to be trained
        optimizer: The optimizer to be used for training
        train_args: The TrainingArgs object containing the training arguments
        epochs: The number of epochs to train for
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

    def freeze_layers(self, freeze: list[str] | int = None):
        """
        Freezes layers of YOLO, always freezes dfl layers
        
        Args:
            freeze: A list of layer names to freeze, or an integer to freeze the first n layers.
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

    @staticmethod
    def label_loss_items(loss_items: torch.Tensor, loss_types: list[str], step: str = "VAL") -> dict[str, float] | list[str]:
        """
        Returns a loss dict with labelled training loss items tensor.
        Not needed for classification but necessary for segmentation & detection.
        Taken from:
        https://github.com/ultralytics/ultralytics/blob/09a34b19eddda5f1a92f1855b1f25f036300d9a1/ultralytics/engine/trainer.py#L628

        Args:
            loss_items: A tensor containing the loss items
            loss_types: A list of loss type names
            step: the prefix to prepend to the loss name/key

        Returns:
        loss_dict if loss_items is not None, else a list of loss names
        """
        keys = [f"{step}/{x}" for x in loss_types]
        if loss_items is not None:
            loss_items = [x for x in loss_items]
            loss_dict = dict(zip(keys, loss_items))
            return loss_dict
        else:
            return keys

    def preprocess_train(
        self, batch: dict[str, Union[torch.Tensor, int, float, str]]
    ):  # pragma: no cover
        """
        Preprocesses a batch of images by scaling and converting to float.
        Code adapted from preprocess_batch method in: ultralytics/models/yolo/detect/train.py
        Note: We didn't include the self.args.multi_scale if statement from the source code

        Args: 
            batch: A dictionary containing the batch of images to be processed

        Return:
            A dictionary containing the preprocessed batch of images
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def preprocess_val(
        self, batch: dict[str, Union[torch.Tensor, int, float, str]]
    ):  # pragma: no cover
        """
        Preprocesses a batch of images for validation.
        Code adapted from preprocess method in: ultralytics/models/yolo/detect/val.py
        Note: We didn't include the self.args.save_hybrid if statement from the source code
        TODO: Test that values are noramlized between (0-1 for example)
        Args: 
            batch: A dictionary containing the batch of images to be processed

        Return:
            A dictionary containing the preprocessed batch of images
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (
            batch["img"].half() if self.args.half else batch["img"].float()
        ) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch

    def training_step(self, minibatch: Union[torch.Tensor, int, float, str]) -> dict:
        """
        Performs a training step on the model given a minibatch of data.
        Args: 
            minibatch: A dictionary containing the batch of images and labels (bounding boxes)

        Return:
            A tuple containing the loss values and loss items
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
        """
        Args: 
            loss_items: A tensor containing the loss items
            step: The prefix to prepend to the loss name/key

        Return:
            A dictionary containing the loss values and validation metric values
        """
        metrics, val_loss = self.validator(self.model, loss_items)
        num_batches = len(self.validator.dataloader)
        losses = self.label_loss_items(val_loss.cpu() / num_batches, self.loss_types, step)
        metrics = {**metrics, **losses}
        return metrics

    def get_signature(self, dataloader: DataLoader) -> ModelSignature:  # pragma: no cover
        """
        Returns the mlflow signature of the model for logging and registration

        Args:
         dataloader: A dataloader containing data to infer the signature
        
        Returns:
            ModelSignature: MLflow ModelSignature object

        NOTE: the function (infer_signature) used below is tested by MLFlow:
        https://github.com/mlflow/mlflow/blob/367f3901d5ed5195c71dbea8434dcb7029fe7b78/tests/models/test_signature.py
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

    @staticmethod
    def get_model(model_yaml: str, model_pt: str) -> DetectionModel:
        """
        Function for creating a DetectionModel object based on pretrained model weights
        and yaml file.
        See DetectionTrainer class and BaseTrainer class for details on how to setup the model

        Args:
            model_yaml: str, path to yaml file for YOLO model
            model_pt: str, path to pretrained YOLO model weights

        Returns:
            DetectionModel, an Ultralytics pytorch model with pretrained weights and class names attached
        """
        # get params for model from: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
        args = ultralytics.cfg.get_cfg()
        model = DetectionModel(cfg=model_yaml, verbose=False)
        weights, _ = attempt_load_one_weight(model_pt)
        model.load(weights)
        model.nc = 1  # attach number of classes to model
        model.names = {0: "ct"}  # attach class names to model
        model.args = args
        # Note that this isn't set in cfg/default.yaml so must set it ourselves
        model.args.conf = 0.002
        # Set to true for towerscout since there's only 1 class
        model.args.single_cls = True

        return model
