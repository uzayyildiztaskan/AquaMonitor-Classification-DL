from torchvision.models import resnet50
from .base_model import BaseModel
import torch.nn as nn

class ResNet(BaseModel):
    def __init__(self, num_classes=31):
        super().__init__(num_classes)
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)