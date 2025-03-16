import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class TrainingManager:
    def __init__(self, model, optimizer, criterion, device, output_dir="outputs", scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1s = []
        self.val_f1s = []
        self.best_f1 = 0
        
        self.output_dir = output_dir
        self.is_colab = self._detect_colab()
        self.scheduler = scheduler
        self.current_phase = 1
        self.current_stage = 1
        
        os.makedirs(self.output_dir, exist_ok=True)
        if self.is_colab:
            self.drive_dir = '/content/drive/MyDrive/project_outputs'
            os.makedirs(self.drive_dir, exist_ok=True)

    def _detect_colab(self):
        return 'COLAB_GPU' in os.environ

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'phase': self.current_phase,
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_f1s': self.train_f1s,
            'val_f1s': self.val_f1s
        }
        
        filename = f'phase{self.current_phase}_stage{self.current_stage}_epoch_{epoch}.pt'
        best_name = f'best_phase{self.current_phase}_stage{self.current_stage}.pt'
        
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, best_name))
        
        if self.is_colab:
            torch.save(checkpoint, os.path.join(self.drive_dir, filename))
            if is_best:
                torch.save(checkpoint, os.path.join(self.drive_dir, best_name))

    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Phase {self.current_phase}-{self.current_stage} Training"):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            epoch_loss += loss.item()
            
        train_acc = accuracy_score(all_labels, all_preds) * 100
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        return epoch_loss/len(train_loader), train_acc, train_f1

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
                
        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        return val_loss/len(val_loader), val_acc, val_f1

    def train_phase(self, train_loader, val_loader, epochs=10, phase=1, stage=1):
        self.current_phase = phase
        self.current_stage = stage
        print(f"\n=== Phase {phase} Stage {stage} Training ===")
        
        for epoch in range(epochs):

            train_loss, train_acc, train_f1 = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.train_f1s.append(train_f1)
            
            val_loss, val_acc, val_f1 = self._validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1s.append(val_f1)

            if self.scheduler and phase == 1:
                self.scheduler.step(val_f1)

            is_best = val_f1 > self.best_f1
            if is_best:
                self.best_f1 = val_f1
                
            self._save_checkpoint(epoch, is_best)
            
            print(f"[Phase {phase}.{stage}] Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.1f}% | F1: {val_f1:.4f}")
            
            if self.is_colab:
                from IPython.display import clear_output
                clear_output(wait=True)
            
            self._plot_metrics()
        
        print(f"Phase {phase}.{stage} Complete | Best Val F1: {self.best_f1:.4f}")

    def _plot_metrics(self):
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title(f'Loss (Phase {self.current_phase}.{self.current_stage})')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(self.train_accuracies, label='Train Acc')
        plt.plot(self.val_accuracies, label='Val Acc')
        plt.title(f'Accuracy (Phase {self.current_phase}.{self.current_stage})')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(self.train_f1s, label='Train F1')
        plt.plot(self.val_f1s, label='Val F1')
        plt.title(f'F1 Scores (Phase {self.current_phase}.{self.current_stage})')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                               f'training_metrics_p{self.current_phase}s{self.current_stage}.png'))
        plt.close()

    def update_optimizer(self, new_optimizer, new_scheduler):
        """Update optimizer for new phase/stage"""
        self.optimizer = new_optimizer
        self.best_f1 = 0 
        if new_scheduler:
            self.scheduler = new_scheduler