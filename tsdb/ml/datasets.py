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
import ultralytics.utils as uutils
import ultralytics.data.augment as aug

from tsdb.ml.types import Hyperparameters, ImageMetadata
from tsdb.preprocessing.images import get_image_metadata


class ImageBinaryDataset(Dataset):
    def __init__(self, images): 
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> ImageMetadata:
         return get_image_metadata(self.images[index])


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
        transfrom: Whether to apply the data augmentation transforms (should be True only for training set)
        mosaic_crop_size: size of sub-images in the mosaic image created by the mosaic augmentation object

        NOTE: mosaic_crop_size determines the size of the images *comprising* the mosaic.
        So for a mosaic_crop_size of m the resulting mosiac image will be a size of 2m x 2m 
        since we set n=4 in the Mosaic constructor.
    """

    def __init__(
        self,
        remote: str,
        local: str,
        shuffle: bool,
        hyperparameters: Hyperparameters,
        transform: bool,
        mosaic_crop_size: int,
        **kwargs,
    ):  # pragma: no cover
        super().__init__(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=hyperparameters.batch_size,
            **kwargs,
        )

        format = aug.Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
            batch_idx=False,
            mask_ratio=4,
            mask_overlap=True,
            bgr=0.0,  # Always return BGR, not RGB. Value from default.yaml
        )

        if transform:
            mosaic_aug = ModifiedMosaic(
                self, image_size=mosaic_crop_size, p=hyperparameters.prob_mosaic, n=4
            )

            albumentation = aug.Albumentations(p=1.0)

            rand_flips = [
                aug.RandomFlip(p=hyperparameters.prob_H_flip, direction="horizontal"),
                aug.RandomFlip(p=hyperparameters.prob_V_flip, direction="vertical"),
            ]

            # apply transformations in the same order Ultralyitcs does
            self.transforms = ultralytics.data.augment.Compose(
                [mosaic_aug, albumentation] + rand_flips
            )
            self.transforms.append(format)

        else:
            self.transforms = ultralytics.data.augment.Compose([format])

    def __getitem__(self, index: int) -> Any:  # pragma: no cover
        """
        Get and return a single sample from the dataset.

        Args:
            index: Index of the sample to return
        Returns:
            A dictionary containing the (potentially transformed) image and label information
        """
        labels = self.get_image_and_label(index)
        return self.transforms(labels)

    def get_image_and_label(self, index: int) -> Any:
        """
        Get and return image & label information from the dataset.
        This function is added because the Ultralytics augmentation objects
        like Mosaic require the dataset object to have a method called
        get_image_and_label.
        
        Args:
            index: Index of the sample to return
        
        Returns:
            A dictionary containing the image and label information
        """
        label = super().__getitem__(index)  # get row from dataset

        # reshape bboxes array from shape (N*4,) to (N, 4)
        # where N is the number of boxes
        bboxes = deepcopy(label.pop("bboxes")).reshape(-1, 4)

        # convert bboxes from xyxy to xywh
        bboxes = uutils.ops.xyxy2xywh(bboxes)

        # move bboxes from "bboxes" key to "instances" key
        instances = uutils.instance.Instances(
            bboxes=bboxes,
            # See: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py#L227
            # We set this value as the calcuations some class methods of Instance perform
            # require it to not be the default value of None
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        )

        # rename "image_path" key to "im_file"
        label["im_file"] = label.pop("image_path")

        # make a deepcopy to make cls array writeable otherwise
        # this line in Format class causes a warning:
        # https://github.com/ultralytics/ultralytics/blob/09a34b19eddda5f1a92f1855b1f25f036300d9a1/ultralytics/data/augment.py#L2058
        label["cls"] = deepcopy(label.pop("cls"))
        label["instances"] = instances
        return label


class ModifiedMosaic(aug.Mosaic):
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

    def __init__(self, dataset: Dataset, image_size: int, p: float, n: int):  # pragma: no cover
        super().__init__(dataset=dataset, imgsz=image_size, p=p, n=n)

    def get_indexes(self) -> list[int]:  # pragma: no cover
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.
        This implementation removes the 'buffer' parameter and always uses the entire dataset

        This method selects random image indexes from the entire dataset.
        It is used to choose images for creating mosaic augmentations.

        Returns:
            A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).
        """
        # select from any images in dataset
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]


def collate_fn_img(data: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Function for collating data into batches. Some additional
    processing of the data is done in this function as well
    to get the batch into the format expected by the Ultralytics
    DetectionModel class.

    Args:
        data: The data to be collated a batch
    Returns: A dictionary containing the collated data in the formated
            expected by the Ultralytics DetectionModel class
    """
    result = defaultdict(list)
    for index, element in enumerate(data):
        # set ratio_pad key to None to have
        # Ultralytics methods compute it for us
        result["ratio_pad"].append(None)

        # This accounts for null images (images with no cooling towers)
        if len(element["cls"]) == 0:
            result["img"].append(element["img"])
            result["im_file"].append(element["im_file"])
            result["ori_shape"].append(element["ori_shape"])
            result["resized_shape"].append(element["resized_shape"])
            continue
        
        else:
            bboxes = element.pop("bboxes")
            result["bboxes"].append(bboxes)

            for key, value in element.items():
                result[key].append(value)

            num_boxes = len(element["cls"])
            for _ in range(num_boxes):
                # yolo dataloader has this as a float not int
                result["batch_idx"].append(float(index))

    # NOTE: No need to call permute here because the Format augmentatoin
    # takes care of this for us
    result["img"] = torch.stack(result["img"], dim=0)
    result["batch_idx"] = torch.tensor(result["batch_idx"])

    # Shape of resulting tensor should be (num bboxes in batch, 4)
    result["bboxes"] = torch.tensor(np.concatenate(result["bboxes"])).reshape(-1, 4)

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
    transform: bool = False,
    mosaic_crop_size: int = 320,
    **kwargs,
) -> DataLoader:
    """
    Function that creates a PyTorch DataLoader from a collection of `.mds` files.

    Args:
    local_dir: Local directory where dataset is cached during training
    remote_dir: The local or remote directory where dataset `.mds` files are stored
    hyperparams: Dataclass containting the batch size of the dataloader and dataset
          See: https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/faqs_and_tips.html
          as well as the mosaic augmentation probability.
    transform: Whether to apply data augmentation
    mosaic_crop_size: The size of sub-images created by 2x2 mosaic augmentation

    Returns:
    A PyTorch DataLoader object
    """

    # NOTE: We do not need to use ToTensor because the Format
    # augmentation object converts our images to tensors for us

    # Note that StreamingDataset is unit tested here:
    # https://github.com/mosaicml/streaming/blob/main/tests/test_streaming.py

    # 
    dataset = YoloDataset(
        local=local_dir,
        remote=remote_dir,
        shuffle=True,
        hyperparameters=hyperparams,
        transform=transform,
        mosaic_crop_size=mosaic_crop_size,
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
    test: DataLoader = None

    @classmethod
    def from_mds(cls, cache_dir: str, mds_dir: str, hyperparams: Hyperparameters, include_test_set: bool = False):  # pragma: no cover
        """
        Class method to create a DataLoaders object from exisiting mds files in a directory with subdirectories
        train, val and test.
        
        Args:
            cache_dir: Directory to store cache files
            mds_dir: Directory where mds files are stored
            hyperparams: Hyperparameters object
        
        NOTE: No testing needed as get_dataloader is already tested.
        """
        clean_stale_shared_memory()
        if include_test_set:
            splits = ("train", "val", "test")
        else:
            splits = ("train", "val")

        dataloaders = [
            get_dataloader(
                local_dir=f"{cache_dir}/{split}",
                remote_dir=mds_dir,
                hyperparams=hyperparams,
                split=split,
                transform=split == "train",
            )
            for split in splits
        ]

        return cls(*dataloaders)