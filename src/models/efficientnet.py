import timm
from .base_model import BaseModel
import torch.nn as nn

class EfficientNetB3(BaseModel):
    def __init__(self, num_classes=31):
        super().__init__(num_classes)
        self.model = timm.create_model("efficientnet_b3", pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)