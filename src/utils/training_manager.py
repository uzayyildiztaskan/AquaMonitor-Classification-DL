import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

class TrainingManager:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(train_loader)
    
    def train(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            val_preds, val_labels = self.model.validate(self.model, val_loader, self.device)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            self.val_metrics.append(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
            
        self._plot_metrics()
    
    def _plot_metrics(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_metrics, label='Validation F1')
        plt.title("Validation Metrics")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.tight_layout()
        plt.show()