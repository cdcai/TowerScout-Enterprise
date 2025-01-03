from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader

from streaming.base.util import clean_stale_shared_memory
from streaming import StreamingDataset

import ultralytics
from ultralytics.utils.instance import Instances
from ultralytics.data.augment import RandomFlip


# NOTE: We create our own ToTensor class since the one by Ultralytics
# doesn't take input data in the same format as the RandomFlip 
# transformation does i.e. a dict with keys "img" and "instances".
class ToTensor:
    def __init__(self, half: bool = False):
        """
        Initializes the ToTensor object for converting images to PyTorch tensors.

        This class is designed to be used as part of a transformation pipeline for image preprocessing in the
        Ultralytics YOLO framework. It converts numpy arrays or PIL Images to PyTorch tensors, with an option
        for half-precision (float16) conversion.
        """
        super().__init__()
        self.half = half

    def __call__(self, labels: dict[str, np.ndarray|Instances]) -> dict[str, torch.Tensor|Instances]:
        labels["img"] = torch.tensor(np.ascontiguousarray(labels["img"]))
        labels["img"] = labels["img"].half() if self.half else labels["img"].float()
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


def collate_fn_img(data: list[dict[str, Any]], transforms: callable) -> dict[str, Any]:
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

    TODO: DELETE HERE
    """
    result = defaultdict(list)

    for index, element in enumerate(data):
        for key, value in element.items():
            if key == "img":
                instances = Instances(
                    bboxes=element["bboxes"].reshape(-1, 4),
                    segments=np.array(
                        [[[5, 5], [10, 10]], [[15, 15], [20, 20]]]
                    ),  # dummy arg, remove later
                    bbox_format="xyxy",
                    normalized=True,
                )
                labels = {"img": value, "instances": instances}
                labels = transforms(labels)
                bboxes = labels["instances"]._bboxes.bboxes
                img = labels["img"]
                result["bboxes"].append(bboxes)
                result["img"].append(img)

                # height & width after some transform/augmentation has been done
                _, height, width = labels["img"].shape
                result["resized_shape"].append((height, width))

                # from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py#L295
                result["ratio_pad"].append(
                    (
                        result["resized_shape"][index][0] / element["ori_shape"][0],
                        result["resized_shape"][index][1] / element["ori_shape"][1],
                    )
                )
            elif key == "bboxes":
                continue
            else:
                result[key].append(value)

        num_boxes = len(element["cls"])

        for _ in range(num_boxes):
            result["batch_idx"].append(
                float(index)
            )  # yolo dataloader has this as a float not int

    result["img"] = torch.stack(result["img"], dim=0)
    result["batch_idx"] = torch.tensor(result["batch_idx"])

    # Shape of resulting tensor should be (num_bboxes_in_batch, 4)
    result["bboxes"] = torch.tensor(np.concatenate(result["bboxes"]))

    # Shape of resulting tensor should be (num_bboxes_in_batch, 1)
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
    batch_size: int,
    transforms: list[callable] = None,
    **kwargs,
) -> DataLoader:
    """
    Function that creates a PyTorch DataLoader from a collection of `.mds` files.

    Args:
    local_dir: Local directory where dataset is cached during training
    remote_dir: The local or remote directory where dataset `.mds` files are stored
    batch_size: The batch size of the dataloader and dataset.
          See: https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/faqs_and_tips.html
    transforms: A list of torchvision transforms to be composed and applied to the images
    Returns:
    A PyTorch DataLoader object

    TODO: DELETE HERE
    """

    # Note that StreamingDataset is unit tested here: https://github.com/mosaicml/streaming/blob/main/tests/test_streaming.py
    dataset = StreamingDataset(
        local=local_dir,
        remote=remote_dir,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    if transforms is None:
       transform = ultralytics.data.augment.Compose([ToTensor()])  
    else:
       transforms = transforms + [ToTensor()]
       transform = ultralytics.data.augment.Compose(transforms)

    # Create PyTorch DataLoader
    collate_fn = partial(collate_fn_img, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
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
    def from_mds(cls, cache_dir: str, mds_dir: str, batch_size: int, transforms=None):
        clean_stale_shared_memory()

        dataloaders = [
            get_dataloader(
                local_dir=f"{cache_dir}/{split}",
                remote_dir=mds_dir,
                batch_size=batch_size,
                split=split,
                transforms=transforms if split == "train" else None,
            )
            for split in ("train", "val", "test")
        ]

        return cls(*dataloaders)