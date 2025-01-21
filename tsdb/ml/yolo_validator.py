import torch
from torch.utils.data import DataLoader

from ultralytics.utils.ops import Profile
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils import IterableSimpleNamespace
from tsdb.ml.model_trainer import BaseTrainer


class ModifiedDetectionValidator(DetectionValidator):
    """
    A modified DetectionValidator object that inherets from the DetectionValidator class from Ultralytics.
    We overload the init_metric method to remove the `val` variable creation and the __call__ method
    to remove componenets that are not relevant to our use case.

    Args:
            dataloader: A PyTorch dataloader for validation/testing datasets.
            args: A IterableSimpleNamespace object containing the arguments for the model and training (if applicable).
            training: Wether the model is being trained or not. This determines some behaviour in __call__.
            device: The device to run the model on.
    """

    def __init__(self, dataloader: DataLoader, args: IterableSimpleNamespace, training: bool, device: str):  # pragma: no cover
        super().__init__(dataloader=dataloader, args=args)
        self.training = training
        self.device = device

    def init_metrics(self, model: torch.nn.Module) -> None:
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
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    @torch.no_grad()
    def __call__(
        self, model: torch.nn.Module, loss_items: torch.Tensor = None
    ) -> dict[str, float] | tuple[dict[str, float], torch.Tensor]:
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
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )

        self.init_metrics(de_parallel(model))
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