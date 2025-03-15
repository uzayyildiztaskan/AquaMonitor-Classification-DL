from torchvision.models import resnet50
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        self.base_model = resnet50(pretrained=True)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

    def unfreeze_last_layers(self, num_layers=10):

        children = list(self.base_model.children())[:-1]
        
        reversed_children = reversed(children)
        
        unfrozen = 0
        
        for child in reversed_children:
            if unfrozen >= num_layers:
                break
            for param in child.parameters():
                param.requires_grad = True
                unfrozen += 1
                if unfrozen >= num_layers:
                    break

    def forward(self, x):
        return self.base_model(x)