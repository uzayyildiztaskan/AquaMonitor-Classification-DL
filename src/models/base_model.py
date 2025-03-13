import torch.nn as nn
from abc import ABC, abstractmethod
import torch

class BaseModel(ABC, nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        self.num_classes = num_classes
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @staticmethod
    def validate(model, val_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return all_preds, all_labels