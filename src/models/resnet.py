from torchvision.models import resnet50
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes=31, class_weights=None):
        super().__init__()
        self.base_model = resnet50(pretrained=True)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        self.weight_decay = 1e-4

        self.class_weights = class_weights


    def unfreeze_last_layers(self, num_layers=10):
        layers = []
        for name, child in self.base_model.named_children():
            if name != 'fc':
                layers.append(child)
        
        reversed_layers = reversed(layers)
        
        unfrozen = 0
        for layer in reversed_layers:
            if unfrozen >= num_layers:
                break
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen += 1
                if unfrozen >= num_layers:
                    break

    def forward(self, x):
        return self.base_model(x)