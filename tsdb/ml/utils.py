from dataclasses import dataclass
from typing import TypedDict
from enum import Enum, auto

import mlflow


class Steps(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


@dataclass
class UCModelName:
    """
    A class to represent the full name of a model registered in
    Unity Catalog

    Attributes:
        catalog: The catalog the model is under
        schema: The schema the model is under
        name: The name of the model in Unity Catalog
    """

    catalog: str
    schema: str
    name: str

    def __str__(self): 
        return f"{self.catalog}.{self.schema}.{self.name}"


class YOLOv5Detection(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_: int
    class_name: str
    secondary: int


def cut_square_detection(img, x1, y1, x2, y2):
    w,h = img.size

    # first, convert detection fractional coordinates into pixels
    x1 *= w
    x2 *= w
    y1 *= h
    y2 *= h

    # compute width and height of cut area
    wc = x2-x1
    hc = y2-y1
    size = int(max(wc,hc)*1.5+(25*640/w)) # 25 is adjusted by image size (Google is 1280, Bing 640)

    # now square it
    cy = (y1+y2)/2.0
    y1 = cy - size/2.0
    y2 = cy + size/2.0
    
    cx = (x1+x2)/2.0
    x1 = cx - size/2.0
    x2 = cx + size/2.0

    # clip to picture
    x1 = max(0,x1)
    x2 = min(w,x2)
    y1 = max(0,y1)
    y2 = min(h,y2)

    return img.crop((x1, y1, x2, y2))

def get_model_tags(model_name: str, alias: str) -> tuple[dict[str, str], str]:  # pragma: no cover
    """
    Returns the tags for the model with the given model name and alias
    along with the model version.
    
    Note we do not unit test this function as `get_model_version_by_alias`
    and `get_model_version` are unit tested in the MLflow git repo already at:
    
        - https://github.com/mlflow/mlflow/blob/8c07dc0f604565bec29358526db461ca4f842bb5/tests/tracking/test_client.py#L1532
    
        - https://github.com/mlflow/mlflow/blob/8c07dc0f604565bec29358526db461ca4f842bb5/tests/store/model_registry/test_rest_store.py#L306
    """
    client = mlflow.MlflowClient()
    model_version_info = client.get_model_version_by_alias(
        name=model_name, alias=alias
    )
    model_version = model_version_info.version
    model_version_details = client.get_model_version(
        name=model_name, version=model_version
    )
    model_tags = model_version_details.tags

    return model_tags, model_version
