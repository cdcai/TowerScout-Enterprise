# Databricks notebook source
# MAGIC %pip install efficientnet_pytorch

# COMMAND ----------

from tsdb.ml.models import TowerScoutModel

# COMMAND ----------

import torch
from torch import nn, optim
from torch import Tensor
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
from enum import Enum, auto
from collections import namedtuple

# COMMAND ----------

# MAGIC %reload_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

def transform_row(batch_pd):
    """
    Defines how to transform partition elements
    """
    transformers = [
        torchvision.transforms.Lambda(lambda image: Image.open(io.BytesIO(image)))
    ]

    transformers.extend(
        [
            torchvision.transforms.Resize(128),
            torchvision.transforms.ToTensor(),
        ]
    )

    transformer_pipeline = torchvision.transforms.Compose(transformers)

    # Needs to be row-major array
    batch_pd["features"] = batch_pd["content"].map(
        lambda image: np.ascontiguousarray(transformer_pipeline(image).numpy())
    )

    return batch_pd[["features"]]


def get_transform_spec():
    """
    Applies transforms across partitions
    """
    spec = TransformSpec(
        partial(transform_row),
        edit_fields=[
            ("features", np.float32, (3, 128, 128), False),
        ],
        selected_fields=["features"],
    )

    return spec

# COMMAND ----------

def get_converter(
    cat_name="edav_dev_csels", sch_name="towerscout_test_schema", batch_size=8
):
    petastorm_path = "file:///dbfs/tmp/petastorm/cache"
    images = spark.table(f"{cat_name}.{sch_name}.image_metadata").select(
        "content", "path"
    )

    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, petastorm_path)

    # Calculate bytes
    num_bytes = (
        images.withColumn("bytes", F.lit(4) + F.length("content"))
        .groupBy()
        .agg(F.sum("bytes").alias("bytes"))
        .collect()[0]["bytes"]
    )

    # Cache
    converter = make_spark_converter(
        images, parquet_row_group_size_bytes=int(num_bytes / sc.defaultParallelism)
    )

    context_args = {"transform_spec": get_transform_spec(), "batch_size": 8}

    return converter


# converter, context_args = get_converter()

# with converter.make_torch_dataloader(**context_args) as dataloader:
#     for image in dataloader:
#         plt.imshow(image["features"].squeeze(0).permute(1, 2, 0))
#         break

# COMMAND ----------

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, images):
        images = self.encoder(images)
        return self.decoder(images)

# COMMAND ----------

def set_optimizer(model, optlr=0.0001, optmomentum=0.9, optweight_decay=1e-4):
    params_to_update = []
    for name, param in model.named_parameters():
        if "_bn" in name:
            param.requires_grad = False
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = optim.SGD(
        params_to_update, lr=optlr, momentum=optmomentum, weight_decay=optweight_decay
    )
    return optimizer

# COMMAND ----------

class Metrics(Enum):
    MSE = nn.MSELoss()

class Steps(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

# COMMAND ----------

def score(logits, labels, step: str, metrics: Metrics):
        return {
            f"{metric.name}_{step}": metric.value(logits, labels).cpu().item()
            for metric in metrics
        }

def forward_func(model, minibatch) -> tuple[Tensor,Tensor,Tensor]:
        images = minibatch["features"]
        labels = minibatch["labels"]

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        logits = model(images)

        return logits, images, labels
    

@torch.no_grad()
def inference_step(minibatch, model, metrics, step) -> dict:
    model.eval()
    logits, _, labels = forward_func(model, minibatch)
    return score(logits, labels, step, metrics)

# COMMAND ----------

ModelOutput = namedtuple("ModelOutput", ["loss", "logits", "images"])


class TowerScoutModelTrainer:
    def __init__(self, optimizer_args, metrics=None, criterion: str = "MSE"):
        self.model = TowerScoutModel()

        if metrics is None:
            metrics = [Metrics.MSE]
        self.metrics = metrics

        optimizer = self.get_optimizer()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)

        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        self.loss = 0
        self.val_loss = 0
        self.threshold = 0.5

    @staticmethod
    def get_optimizer():
        return torch.optim.Adam

    # def forward(self, minibatch) -> ModelOutput:
    #     images = minibatch["features"]
    #     labels = minibatch["labels"]

    #     if torch.cuda.is_available():
    #         images = images.cuda()
    #         labels = labels.cuda()

    #     logits = self.model(images)
    #     loss = self.criterion(logits, images)

    #     return ModelOutput(loss, logits, images)

    # def score(self, logits, labels, step: str):
    #     return {
    #         f"{metric.name}_{step}": metric.value(logits, labels).cpu().item()
    #         for metric in self.metrics
    #     }

    def training_step(self, minibatch, **kwargs) -> dict:
        self.model.train()

        logits, images, labels = forward_func(self.model, minibatch)
        loss = self.criterion(logits, images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return score(logits, labels, Steps["TRAIN"].name, self.metrics)

    @torch.no_grad()
    def validation_step(self, minibatch, **kwargs) -> dict:
        return inference_step(minibatch, self.model, self.metrics, Steps["VAL"].name)
        # self.model.eval()
        # output = forward_func(self.model, minibatch)
        # return score(output.logits, output.labels, step, self.metrics)

    def save_model(self):
        pass
