import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

class TrainingManager:
    def __init__(self, model, optimizer, criterion, device, output_dir="outputs"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        self.best_f1 = 0
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Colab detection
        self.is_colab = self._detect_colab()
        if self.is_colab:
            from google.colab import drive
            drive.mount('/content/drive')
            self.drive_dir = '/content/drive/MyDrive/project_outputs'
            os.makedirs(self.drive_dir, exist_ok=True)

    def _detect_colab(self):
        try:
            import os
            return 'COLAB_GPU' in os.environ
        except:
            return False

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best_model.pth'))
        
        if self.is_colab:
            torch.save(checkpoint, os.path.join(self.drive_dir, f'checkpoint_epoch_{epoch}.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(self.drive_dir, 'best_model.pth'))

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
            
            # Save best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self._save_checkpoint(epoch, is_best=True)
            else:
                self._save_checkpoint(epoch)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if self.is_colab:
                from IPython.display import clear_output
                clear_output(wait=True)
            
            self._plot_metrics()
        
        self._save_checkpoint('final')
        print(f"Training complete. Models saved to {self.output_dir}")
        if self.is_colab:
            print(f"Backup copies saved to Google Drive: {self.drive_dir}")

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
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
        plt.close()