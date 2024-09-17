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
