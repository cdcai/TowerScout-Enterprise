# Databricks notebook source
import torch
from torch import nn
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet


# COMMAND ----------

def transform_row(batch_pd):
    """
    Defines how to transform partition elements
    """
    transformers = [
        torchvision.transforms.Lambda(lambda image: Image.open(io.BytesIO(image)))
    ]

    transformers.extend([
        torchvision.transforms.Resize(128),
        torchvision.transforms.ToTensor(),
    ])

    transformer_pipeline = torchvision.transforms.Compose(transformers)

    # Needs to be row-major array
    batch_pd["features"] = (
        batch_pd["content"]
        .map(
            lambda image: np.ascontiguousarray(transformer_pipeline(image).numpy())
        )
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
        selected_fields=["features"]
    )

    return spec

# COMMAND ----------

def get_converter(cat_name="edav_dev_csels", sch_name="towerscout_test_schema", batch_size=8):
    petastorm_path = "file:///dbfs/tmp/petastorm/cache"
    images = (
        spark
        .table(f"{cat_name}.{sch_name}.image_metadata")
        .select("content", "path")
    )

    spark.conf.set(
        SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 
        petastorm_path
    )

    # Calculate bytes
    num_bytes = (
        images
        .withColumn(
            "bytes", 
            F.lit(4) + F.length("content")).groupBy().agg(F.sum("bytes").alias("bytes")).collect()[0]["bytes"]
    )

    # Cache
    converter = make_spark_converter(
        images, 
        parquet_row_group_size_bytes=int(num_bytes/sc.defaultParallelism)
    )

    context_args = {
        "transform_spec": get_transform_spec(),
        "batch_size": 8
    }

    return converter

converter, context_args = get_converter()

with converter.make_torch_dataloader(**context_args) as dataloader:
    for image in dataloader:
        plt.imshow(image["features"].squeeze(0).permute(1, 2, 0))
        break

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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        images = self.encoder(images)
        return self.decoder(images)


# COMMAND ----------

class TowerScoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5', include_top=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model._fc = nn.Sequential(
                nn.Linear(2048, 512), #b5
                nn.Linear(512, 1))
        self.model.cuda()
        self.model._fc

# COMMAND ----------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# COMMAND ----------

def set_optimizer(model, optlr=0.0001, optmomentum=0.9, optweight_decay=1e-4):
    params_to_update = []
    for name, param in model.named_parameters():
        if "_bn" in name:
            param.requires_grad = False
        if param.requires_grad == True:
            params_to_update.append(param)
        
    optimizer = optim.SGD(params_to_update,
                        lr=optlr,
                        momentum=optmomentum,
                        weight_decay=optweight_decay)
    return optimizer

# COMMAND ----------

class TowerScoutModelTrainer():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        self.loss = 0
        self.threshold = 0.5
    
    def preprocess_data(self):
        pass

    def training_step(self, minibatch, **kwargs) -> dict[str, float]:
        images, labels = minibatch
        total, correct = 0, 0
        images = images.cuda()
        labels = labels.cuda()
        labels = labels.unsqueeze(1).float()
        
        out = self.model(images)
        self.loss = self.criterion(out, labels)
        
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        total += labels.size(0)
        out = torch.sigmoid(out)
        correct += ((out > self.threshold).int() == labels).sum().item()

        return {"loss": self.loss.item(),
                 correct}

    def validation_step(self):
        pass
    
    def save_model(self):
        pass


# COMMAND ----------


