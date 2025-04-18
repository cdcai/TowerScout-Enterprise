"""This module contains code for validating DetectionModels from Ultralytics"""
from copy import copy

import torch
from torch.utils.data import DataLoader

from pyspark.sql.functions import current_timestamp

import ultralytics.utils as uutils
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionValidator

from tsdb.ml.types import Hyperparameters
from tsdb.ml.datasets import get_dataloader


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

    def __init__(
        self,
        dataloader: DataLoader,
        args: uutils.IterableSimpleNamespace,
        training: bool,
        device: str,
    ):  # pragma: no cover
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
        self.confusion_matrix = uutils.metrics.ConfusionMatrix(
            nc=self.nc, conf=self.args.conf
        )
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


def benchmark_model(
    model: DetectionModel, mds_benchmark_dir: str, cache_dir: str, batch_size: int
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    This function takes a YOLO model from the newer Ultralytics library and benchmarks it against the MDS dataset
    at the directory supplied. To do this we use the ModifiedDetectionValidator object from the tsdb library.

    Args:
        model: the DetectionModel we want to benchmark
        mds_benchmark_dir: the directory containing the MDS files of the benchmark dataset
        cache_dir: the directory for the dataloader to use
        batch_size: the batch size for the dataloader

    Returns:
        overall_metrics: a dictionary of the overall metrics (f1, precision, recall) for the model
        per_class_metrics: a dictionary of the metrics (f1, precision, recall) for each class for the model
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(
        local_dir=cache_dir,
        remote_dir=mds_benchmark_dir,
        hyperparams=Hyperparameters(batch_size=batch_size),
        split=None,
        transform=False,
    )

    validator = ModifiedDetectionValidator(
        dataloader=dataloader, training=False, device=device, args=copy(model.args)
    )

    metrics = validator(model)
    per_class = validator.metrics.box

    overall_metrics = {
        "f1": metrics["metrics/f1(B)"],
        "precision": metrics["metrics/precision(B)"],
        "recall": metrics["metrics/recall(B)"],
    }

    per_class_metrics = {
        class_name: {
            "f1": per_class.f1[class_label],
            "precision": per_class.p[class_label],
            "recall": per_class.r[class_label],
        }
        for class_label, class_name in model.names.items()
    }

    return overall_metrics, per_class_metrics


def update_benchmark_table(
    benchmark_table: str,
    overall_metrics: dict[str, float],
    per_class_metrics: dict[str, dict[str, float]],
    model_metadata: dict,
) -> None:
    """
    This function updates the benchmark table with the supplied overall metrics, per class
    metrics and model metadata for a benchmark run of the model.

    Args:
        benchmark_table: the name of the benchmark table to update in the format:
                         {catalog}.{schema}.{table_name}
        overall_metrics: a dictionary of the overall metrics (f1, precision, recall) for the model
        per_class_metrics: a dictionary of the metrics (f1, precision, recall) for each class for the model
        model_metadata: a dictionary of the model metadata (uc_model_name, uc_model_version, model_uri) to include in the benchmark table
    """

    # must cast to float from numpy.float64 because PySpark FloatType can't take numpy floats.
    overall_metrics = {key: float(value) for key, value in overall_metrics.items()}
    per_class_metrics = {
        key: {k: float(v) for k, v in value.items()}
        for key, value in per_class_metrics.items()
    }

    data = [(model_metadata, None, overall_metrics, per_class_metrics)]

    schema = spark.read.format("delta").table(benchmark_table).schema

    new_row_df = spark.createDataFrame(data, schema)
    new_row_df = new_row_df.withColumn("benchmarked_at", current_timestamp())
    new_row_df.write.format("delta").mode("append").saveAsTable(benchmark_table)