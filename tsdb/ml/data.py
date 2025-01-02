from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

import torch
from torch.utils.data import DataLoader

from streaming.base.util import clean_stale_shared_memory

from ultalytics.utils.instance import Instances
from ultralytics.data.augment import RandomFlip

from tsdb.preprocessing.preprocess import get_dataloader


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
                instances = Instances(bboxes=element["bboxes"].reshape(-1, 4), bbox_format="xyxy", normalized=True)
                labels = {"img": value, "instances": instances}
                labels = transforms(labels)
                bboxes = labels["instances"]._bboxes.bboxes
                img = labels["img"]
                result["bboxes"].append(bboxes)
                result["img"].append(torch.tensor(img))
                
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