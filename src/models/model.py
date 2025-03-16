from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
import torch

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=31, class_weights=None):
        super().__init__()
        base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        for param in base_model.parameters():
            param.requires_grad = False
        
        self.features = base_model.features

        self.classifier = nn.Sequential(
            LayerNorm2d((768,), eps=1e-6),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )
        self.class_weights = class_weights
        self.weight_decay = 3e-5

    def unfreeze_last_layers(self, num_layers=10):
        layers = list(self.features.children())
        unfrozen = 0
        for layer in reversed(layers):
            if unfrozen >= num_layers:
                break
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen += 1
                if unfrozen >= num_layers:
                    break

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model_path = "./outputs_convnext/phase2_stage3_epoch_1.pt"

model_eval = ConvNeXt(31)
checkpoint = torch.load(model_path, map_location='cpu')
model_eval.load_state_dict(checkpoint['model_state_dict'])

