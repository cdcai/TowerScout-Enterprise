from torch import nn
from efficientnet_pytorch import EfficientNet


class TowerScoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b5", include_top=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model._fc = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, 1))  # b5
        if torch.cuda.is_available():
            self.model.cuda()
        self.model._fc

    def forward(self, input):
        return self.model(input)


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