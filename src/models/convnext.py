from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Input shape: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # to [B, H, W, C]
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # back to [B, C, H, W]
        return x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=31, class_weights=None):
        super().__init__()
        # Load pretrained ConvNeXt Tiny
        base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # Freeze base model layers
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Replace classifier entirely
        self.features = base_model.features  # feature extractor

        # Define new classifier manually
        self.classifier = nn.Sequential(
            LayerNorm2d((768,), eps=1e-6),  # Correct version
            nn.AdaptiveAvgPool2d(1),           # Pool to 1x1 spatial
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )
        self.class_weights = class_weights
        self.weight_decay = 3e-5

    def unfreeze_last_layers(self, num_layers=10):
        # Unfreeze last layers of feature extractor
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
        x = self.features(x)  # Pass through backbone
        x = self.classifier(x)  # Pass through custom classifier
        return x
