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
        self.train_accuracies = []
        self.val_losses = [] 
        self.best_f1 = 0
        self.output_dir = output_dir
        self.current_phase = 1  # Track training phase
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_colab = self._detect_colab()
        if self.is_colab:
            from google.colab import drive
            drive.mount('/content/drive')
            self.drive_dir = '/content/drive/MyDrive/project_outputs'
            os.makedirs(self.drive_dir, exist_ok=True)

    def _detect_colab(self):
        return 'COLAB_GPU' in os.environ if 'COLAB_GPU' in os.environ else False

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        # Save with phase information
        filename = f'phase{self.current_phase}_checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, f'phase{self.current_phase}_best_model.pt'))
        
        if self.is_colab:
            torch.save(checkpoint, os.path.join(self.drive_dir, filename))
            if is_best:
                torch.save(checkpoint, os.path.join(self.drive_dir, f'phase{self.current_phase}_best_model.pt'))

    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0 
        
        for batch in tqdm(train_loader, desc=f"Phase {self.current_phase} - Training"):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            epoch_loss += loss.item()
            
        train_acc = 100 * correct / total
        return epoch_loss / len(train_loader), train_acc

    def train_phase(self, train_loader, val_loader, epochs=10, phase=1):
        self.current_phase = phase
        print(f"\n=== Starting Phase {phase} Training ===")
        
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader)  
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            val_loss, val_preds, val_labels = self._validate(val_loader)
            self.val_losses.append(val_loss)
              
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            self.val_metrics.append(val_f1)

            is_best = val_f1 > self.best_f1
            if is_best:
                self.best_f1 = val_f1
                
            self._save_checkpoint(epoch, is_best)
            
            print(f"Phase {phase} - Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if self.is_colab:
                from IPython.display import clear_output
                clear_output(wait=True)
            
            self._plot_metrics()
        
        print(f"Phase {phase} complete. Best F1: {self.best_f1:.4f}")

    def _validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        return val_loss/len(val_loader), all_preds, all_labels

    def _plot_metrics(self):
        plt.figure(figsize=(15, 6))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training/Validation Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_metrics, label='Val F1 Score')
        plt.title('Training Accuracy & Validation F1')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
        plt.close()

    def update_optimizer(self, new_optimizer):
        """Update optimizer for new phase"""
        self.optimizer = new_optimizer
        self.best_f1 = 0  # Reset best metric for new phase
        self.train_losses = []
        self.val_metrics = []