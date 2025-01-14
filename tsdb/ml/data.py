from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from copy import deepcopy
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from streaming.base.util import clean_stale_shared_memory
from streaming import StreamingDataset

import ultralytics
from ultralytics.utils.instance import Instances
from ultralytics.data.augment import RandomFlip, Mosaic
from ultralytics.utils.ops import xyxy2xywh

from tsdb.ml.utils import Hyperparameters


class YoloDataset(StreamingDataset):
    """
    A dataset object that inherets from the StreamingDataset class with
    modifications to be compatible with the Ultralytics Mosaic
    (not to be confused with MosaicML and its associated streaming library) augmentation object.
    See https://docs.mosaicml.com/projects/streaming/en/stable/how_to_guides/cifar10.html#Loading-the-Data
    for more details on working with custom MosaicML Streaming datasets.

    Args:
        remote: Remote path or directory to download the dataset from.
        local:  Local working directory to download shards to.
                This is where shards are cached while they are being used.
                Uses a temp directory if not set
        shuffle:  Whether to iterate over the samples in randomized orde
        hyperparameters: Hyperparameters to use (batch size)
        image_size: image size used to create the mosaic augmentation object
    """

    def __init__(
        self,
        remote: str,
        local: str,
        shuffle: bool,
        hyperparameters: Hyperparameters,
        transform: list[callable],
        image_size: int,
        **kwargs
    ):
        super().__init__(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=hyperparameters.batch_size,
            **kwargs
        )

        mosaic_aug = ModifiedMosaic(
            self, image_size=image_size, p=hyperparameters.prob_mosaic, n=4
        )

        # prepend mosaic augmentation to input Ultralytics transforms
        self.transforms = ultralytics.data.augment.Compose([mosaic_aug] + transform)

    def __getitem__(self, index: int) -> Any:
        labels = self.get_image_and_label(index)
        
        # Account for the case where the image has no labels (null image)
        if len(labels["cls"]) < 1:
            return labels
        else:
            return self.transforms(labels)

    def get_image_and_label(self, index: int) -> Any:
        """
        Get and return image & label information from the dataset.
        This function is added because the Ultralytics augmentation objects
        like Mosaic require the dataset object to have a method called
        get_image_and_label.
        (We add this to align with the protocol of the Ultralytics Mosiac object)
        """
        label = super().__getitem__(index)  # get row from dataset

        bboxes = deepcopy(label.pop("bboxes")).reshape(-1, 4)
        bboxes = xyxy2xywh(bboxes)
        # move bboxes from "bboxes" key to "instances" key
        instances = Instances(
            bboxes=bboxes,
            # See: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py#L227
            # We set this value as the calcuations some class methods of Instance perform 
            # require it to not be the default value of None
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        )
        # from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py#L295
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation

        # rename "image_path" key to "im_file"
        label["im_file"] = label.pop("image_path")
        label["instances"] = instances
        return label


class ModifiedMosaic(Mosaic):
    """
    A modified Mosaic augmentation object that inherets from the Mosaic class from Ultralytics.
    The sole modification is the removal of the 'buffer' parameter from the Mosaic class
    so that the 'buffer' is always the entire dataset.

    Args:
            dataset: The dataset on which the mosaic augmentation is applied.
            image_size: Image size (height and width) after mosaic pipeline of a single image.
            p: Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n: The grid size, either 4 (for 2x2) or 9 (for 3x3).
    """

    def __init__(self, dataset: Dataset, image_size: int, p: float, n: int):
        """
        NOTE: image_size determines the size of the images *comprising* the mosaic.
        So for an image size of DxD and for a mosaic of 4 images (2 x 2 grid of images)
        the output mosaic image will have a size of 2D x 2D.
        """
        super().__init__(dataset=dataset, imgsz=image_size, p=p, n=n)

    def get_indexes(self) -> list[int]:
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.
        This implementation removes the 'buffer' parameter and always uses the entire dataset

        This method selects random image indexes from the entire dataset.
        It is used to choose images for creating mosaic augmentations.

        Returns:
            (List[int]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        # select from any images in dataset
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]


class ToTensor:
    """
    NOTE: We create our own ToTensor class since the one by Ultralytics
    doesn't take input data in the same format as the RandomFlip
    transformation does i.e. a dict with keys "img" and "instances".
    """

    def __init__(self):
        """
        Initializes the ToTensor object for converting images to PyTorch tensors.

        This class is designed to be used as part of a transformation pipeline for image preprocessing in the
        Ultralytics YOLO framework.
        """
        super().__init__()

    def __call__(
        self, labels: dict[str, np.ndarray | Instances]
    ) -> dict[str, torch.Tensor | Instances]:
        labels["img"] = torch.tensor(np.ascontiguousarray(labels["img"]))
        labels["img"] = labels["img"].to(torch.uint8)
        return labels


def data_augmentation(
    prob_H_flip: float = 0.2,
    prob_V_flip: float = 0.2,
) -> list:
    """
    Data Augmentation function to add label invariant transforms to training pipeline
    Applies a series of transformations such as rotation, horizontal and vertical flips, and Gaussian blur to each image

    TODO: test this
    """
    transforms = [
        RandomFlip(p=prob_H_flip, direction="horizontal"),
        RandomFlip(p=prob_V_flip, direction="vertical"),
    ]
    return transforms


def collate_fn_img(data: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Function for collating data into batches. Some additional
    processing of the data is done in this function as well
    to get the batch into the format expected by the Ultralytics
    DetectionModel class.

    Args:
        data: The data to be collated a batch
        transforms: Ultralytics transforms applied to the np array images
    Returns: A dictionary containing the collated data in the formated
            expected by the Ultralytics DetectionModel class
    """
    result = defaultdict(list)

    for index, element in enumerate(data):
        # This accounts for null images 
        # (images with no cooling towers)
        if len(element["cls"]) < 1:
           result["img"].append(element["img"])
           result["im_file"].append(element["im_file"])
           result["ori_shape"].append(element["ori_shape"])
           result["resized_shape"].append(element["resized_shape"])
           continue

        #element["instances"].convert_bbox(format="xyxy")
        w, h = 640, 640
        #element["instances"].denormalize(w, h)
        bboxes = element.pop("instances")._bboxes.bboxes
        result["bboxes"].append(bboxes)

        for key, value in element.items():
            result[key].append(value)

        num_boxes = len(element["cls"])
        for _ in range(num_boxes):
            # yolo dataloader has this as a float not int
            result["batch_idx"].append(float(index))

    # Reshape tensor from BHWC to BCHW with permute
    # because model expected channel first format
    # however the mosaic augmentation expects channel last
    # hence the reshape is done here.
    result["img"] = torch.stack(result["img"], dim=0).permute(0,3,1,2)
    result["batch_idx"] = torch.tensor(result["batch_idx"])

    # Shape of resulting tensor should be (num bboxes in batch, 4)
    result["bboxes"] = torch.tensor(np.concatenate(result["bboxes"]))

    # Shape of resulting tensor should be (num bboxes in batch, 1)
    # Call np.concatenate to avoid calling tensor on a list of np arrays but instead just one 2d np array
    result["cls"] = torch.tensor(np.concatenate(result["cls"])).reshape(-1, 1)

    result["im_file"] = tuple(result["im_file"])
    result["ori_shape"] = tuple(result["ori_shape"])
    result["resized_shape"] = tuple(result["resized_shape"])
    result["ratio_pad"] = tuple(result["ratio_pad"])

    return dict(result)


def get_dataloader(
    local_dir: str,
    remote_dir: str,
    hyperparams: Hyperparameters,
    transforms: list[callable] = None,
    **kwargs,
) -> DataLoader:
    """
    Function that creates a PyTorch DataLoader from a collection of `.mds` files.

    Args:
    local_dir: Local directory where dataset is cached during training
    remote_dir: The local or remote directory where dataset `.mds` files are stored
    hyperparams: Dataclass containting he batch size of the dataloader and dataset
          See: https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/faqs_and_tips.html
          as well as the mosaic augmentation probability.
    transforms: A list of torchvision transforms to be composed and applied to the images
    Returns:
    A PyTorch DataLoader object

    TODO: DELETE HERE
    """

    if transforms is None:
        transform = [ToTensor()]
    else:
        transform = transforms + [ToTensor()]

    # Note that StreamingDataset is unit tested here:
    # https://github.com/mosaicml/streaming/blob/main/tests/test_streaming.py
    dataset = YoloDataset(
        local=local_dir,
        remote=remote_dir,
        shuffle=True,
        hyperparameters=hyperparams,
        transform=transform,
        image_size=320,
        **kwargs,
    )

    # Create PyTorch DataLoader
    dataloader = DataLoader(
        dataset, batch_size=hyperparams.batch_size, collate_fn=collate_fn_img
    )
    return dataloader


@dataclass
class DataLoaders:
    """
    A dataclass to hold the dataloaders for the training, testing
    and validation sets

    Attributes:
        train: The dataloader for the training dataset
        val: The dataloader for the validation dataset
        test: The dataloader for the testing dataset
    """

    train: DataLoader
    val: DataLoader
    test: DataLoader

    @classmethod
    def from_mds(cls, cache_dir: str, mds_dir: str, hyperparams: Hyperparameters, transforms=None):
        clean_stale_shared_memory()

        dataloaders = [
            get_dataloader(
                local_dir=f"{cache_dir}/{split}",
                remote_dir=mds_dir,
                hyperparams=hyperparams,
                split=split,
                transforms=transforms if split == "train" else None,
            )
            for split in ("train", "val", "test")
        ]

        return cls(*dataloaders)