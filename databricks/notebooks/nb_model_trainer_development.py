# Databricks notebook source
import torch
from torch import nn
from torchvision import transforms, datasets

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

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# COMMAND ----------

class TowerScoutModelTrainer():
    def __init__(self):
        pass
    
    def preprocess_data(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass
    
    def save_model(self):
        pass

